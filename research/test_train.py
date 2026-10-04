"""Tests for research.train — a stub stands in for lerobot-train; no GPU, no lerobot."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import research.train as train
from research.train import METHODS, main, missing_quantile_stats, parse_metrics_line, resolve_step

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real-format lerobot lines (utils/utils.py init_logging + MetricsTracker.__str__).
LOG_500 = "INFO 2026-10-04 12:00:00 ot_train.py:641 step:500 smpl:500 ep:1 epch:0.04 loss:0.812 grdn:3.210 lr:2.5e-05 updt_s:0.812 data_s:0.004 smp/s:1 mem_gb:21.33"
LOG_1K = "INFO 2026-10-04 12:05:00 ot_train.py:641 step:1K smpl:1K ep:2 epch:0.08 loss:0.640 grdn:2.100 lr:5.0e-05 updt_s:0.800 data_s:0.004 smp/s:1 mem_gb:21.33"
LOG_1500 = "INFO 2026-10-04 12:10:00 ot_train.py:641 step:2K smpl:2K ep:3 epch:0.13 loss:0.501 grdn:1.900 lr:5.0e-05 updt_s:0.800 data_s:0.004 smp/s:1 mem_gb:21.33"
LOG_2000 = "INFO 2026-10-04 12:15:00 ot_train.py:641 step:2K smpl:2K ep:4 epch:0.17 loss:0.433 grdn:1.700 lr:4.9e-05 updt_s:0.800 data_s:0.004 smp/s:1 mem_gb:21.33"
NOISE = [
    "INFO 2026-10-04 11:59:00 ot_train.py:441 cfg.steps=2000 (2K)",
    "INFO 2026-10-04 12:10:00 ot_train.py:673 step 1500: eval_loss=0.5123",
    "INFO 2026-10-04 12:10:00 ot_train.py:687 Checkpoint policy after step 1500",
    "Training:  25%|##5       | 500/2000 [01:00<03:00,  8.3step/s]",
    "",
]


def write_stub(tmp_path: Path, exit_code: int = 0, lines=(LOG_500, LOG_1K, LOG_1500, LOG_2000),
               sleep_s: float = 0.0) -> str:
    """Write a fake lerobot-train; return a --lerobot-bin value that runs it."""
    stub = tmp_path / "stub_lerobot_train.py"
    argv_file = tmp_path / "stub_argv.json"
    stub.write_text(textwrap.dedent(f"""
        import json, sys, time
        json.dump(sys.argv[1:], open({str(argv_file)!r}, "w"))
        for line in {list(NOISE[:2]) + list(lines)!r}:
            print(line, file=sys.stderr, flush=True)  # lerobot logs to stderr
            time.sleep({sleep_s})
        sys.exit({exit_code})
    """), encoding="utf-8")
    return f'"{sys.executable}" "{stub}"'


QUANTILE_STATS = {f: {"mean": [0.0], "std": [1.0], "q01": [-1.0], "q99": [1.0]}
                  for f in ("observation.state", "action")}


def write_dataset(tmp_path: Path, stats=QUANTILE_STATS) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return root


def base_args(tmp_path: Path, bin_: str, **over) -> list[str]:
    root = tmp_path / "dataset"
    if "dataset-root" not in over and not root.exists():
        write_dataset(tmp_path)
    opts = {"method": "h0", "dataset": "local/pick_place", "dataset-root": str(root),
            "output": str(tmp_path / "run"), "seed": "0", "steps": "2000", "log-freq": "500",
            "lerobot-bin": bin_, **over}
    return [f"--{k}={v}" for k, v in opts.items() if v is not None]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- log-line regex ---

@pytest.mark.parametrize("line,expected", [
    (LOG_500, {"step": "500", "loss": 0.812, "lr": 2.5e-05}),
    (LOG_1500, {"step": "2K", "loss": 0.501, "lr": 5.0e-05}),
    ("\rTraining:  50%|#####     | 1/2 [00:01<00:01]" + LOG_1K, {"step": "1K", "loss": 0.640, "lr": 5.0e-05}),
])
def test_parse_metrics_line_matches(line, expected):
    assert parse_metrics_line(line) == expected


@pytest.mark.parametrize("line", NOISE + [
    "step:200 smpl:200 ep:1 epch:0.01 loss:abc grdn:1.0 lr:1e-5",  # non-numeric loss
    "INFO ot_train.py:272 {'log_freq': 200, 'steps': 30000}",
])
def test_parse_metrics_line_ignores(line):
    assert parse_metrics_line(line) is None


def test_resolve_step_recovers_rounded_steps():
    assert resolve_step("500", 0, 500) == (500, True)
    assert resolve_step("1K", 500, 500) == (1000, True)
    assert resolve_step("2K", 1000, 500) == (1500, True)
    assert resolve_step("2K", 1500, 500) == (2000, True)
    # Inconsistent with prev + log_freq: fall back to the token, flagged inexact.
    assert resolve_step("5K", 1000, 500) == (5000, False)


# --- quantile stats ---

def test_missing_quantile_stats():
    assert missing_quantile_stats(QUANTILE_STATS) == []
    no_q = {f: {"mean": [0.0], "std": [1.0]} for f in ("observation.state", "action")}
    assert missing_quantile_stats(no_q) == ["observation.state", "action"]
    assert missing_quantile_stats({"action": QUANTILE_STATS["action"]}) == ["observation.state"]


def test_dataset_without_quantiles_is_refused(tmp_path, capsys):
    stats = {f: {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.1]}
             for f in ("observation.state", "action")}
    write_dataset(tmp_path, stats)
    rc = main(base_args(tmp_path, write_stub(tmp_path)))
    err = capsys.readouterr().err
    assert rc == 2
    assert "q01/q99" in err and "lerobot-edit-dataset" in err and "--operation.type=recompute_stats" in err
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "stub_argv.json").exists()  # never launched


def test_unreadable_stats_is_refused(tmp_path, capsys):
    rc = main(base_args(tmp_path, write_stub(tmp_path), **{"dataset-root": str(tmp_path / "nowhere")}))
    assert rc == 2
    assert "stats.json" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_hub_dataset_stats_are_fetched(tmp_path, capsys, monkeypatch):
    fetched = []
    monkeypatch.setattr(train, "_read_hub_stats", lambda repo: fetched.append(repo) or QUANTILE_STATS)
    rc = main(base_args(tmp_path, "lerobot-train", **{"dataset-root": None}) + ["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert fetched == ["local/pick_place"]
    assert "STATS ok" in out.splitlines()
    assert "--dataset.root" not in out


def _unreachable_hub(repo):
    raise train.StatsError(f"cannot download meta/stats.json for {repo} (ConnectionError: offline)")


@pytest.mark.parametrize("where", ["local_root", "hub"])
def test_dry_run_with_unreachable_dataset_is_unchecked(tmp_path, capsys, monkeypatch, where):
    if where == "hub":
        monkeypatch.setattr(train, "_read_hub_stats", _unreachable_hub)
        over = {"dataset-root": None}
    else:
        over = {"dataset-root": str(tmp_path / "nowhere")}
    rc = main(base_args(tmp_path, "lerobot-train", **over) + ["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0].startswith("STATS unchecked: ")
    assert "COMMAND lerobot-train " in out
    assert not (tmp_path / "run").exists()


def test_dry_run_with_missing_quantiles_reports_and_exits_0(tmp_path, capsys):
    write_dataset(tmp_path, {f: {"mean": [0.0], "std": [1.0]} for f in ("observation.state", "action")})
    rc = main(base_args(tmp_path, "lerobot-train") + ["--dry-run"])
    out = capsys.readouterr().out
    stats_line = out.splitlines()[0]
    assert rc == 0
    assert stats_line.startswith("STATS missing q01/q99")
    assert "lerobot-edit-dataset" in stats_line and "--operation.type=recompute_stats" in stats_line
    assert "COMMAND lerobot-train " in out
    assert not (tmp_path / "run").exists()


# --- CLI ---

def test_dry_run_prints_command_and_creates_nothing(tmp_path, capsys):
    rc = main(base_args(tmp_path, "lerobot-train") + ["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0] == "STATS ok"
    assert "COMMAND lerobot-train --policy.type=pi05" in out
    assert "--policy.train_expert_only=true" in out
    assert [p.name for p in tmp_path.iterdir()] == ["dataset"]  # only the fixture


def test_missing_seed_fails(tmp_path, capsys):
    args = [a for a in base_args(tmp_path, "lerobot-train") if not a.startswith("--seed=")]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code != 0
    assert "--seed" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_existing_nonempty_output_is_refused_and_untouched(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text('{"status": "done"}', encoding="utf-8")
    rc = main(base_args(tmp_path, write_stub(tmp_path)))
    assert rc == 2
    assert "already exists" in capsys.readouterr().err
    assert [p.name for p in run.iterdir()] == ["result.json"]
    assert (run / "result.json").read_text(encoding="utf-8") == '{"status": "done"}'
    assert not (tmp_path / "stub_argv.json").exists()  # never launched


def test_h0_maps_to_expected_policy_flags(tmp_path, capsys):
    assert main(base_args(tmp_path, write_stub(tmp_path))) == 0
    received = json.loads((tmp_path / "stub_argv.json").read_text(encoding="utf-8"))
    for key, value in METHODS["h0"].items():
        assert f"--{key}={value}" in received
    assert "--policy.train_expert_only=true" in received
    assert "--policy.freeze_vision_encoder=false" in received
    assert "--policy.pretrained_path=lerobot/pi05_base" in received
    assert "--policy.dtype=bfloat16" in received
    assert "--policy.gradient_checkpointing=true" in received
    assert "--policy.push_to_hub=false" in received
    assert f"--dataset.root={tmp_path / 'dataset'}" in received
    assert "--seed=0" in received
    assert f"--output_dir={(tmp_path / 'run' / 'lerobot').as_posix()}" in received


def test_successful_run_writes_bookkeeping(tmp_path, capsys):
    rc = main(base_args(tmp_path, write_stub(tmp_path)))
    out = capsys.readouterr().out.splitlines()
    run = tmp_path / "run"
    assert rc == 0

    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["args"]["seed"] == 0
    assert config["args"]["lr"] == 5e-5 and config["args"]["batch_size"] == 1
    assert config["method_flags"] == METHODS["h0"]
    assert config["lerobot_argv"][-1] == "--wandb.enable=false"
    for key in ("git_commit", "git_dirty", "lerobot_version", "python_version", "hostname", "start_time"):
        assert key in config

    metrics = read_jsonl(run / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [500, 1000, 1500, 2000]
    assert all(m["step_exact"] for m in metrics)
    assert metrics[-1]["loss"] == 0.433

    assert out[0] == "STATUS started"
    assert out[1:5] == [
        "PROGRESS step=500/2000 loss=0.812 lr=2.5e-05",
        "PROGRESS step=1000/2000 loss=0.64 lr=5e-05",
        "PROGRESS step=1500/2000 loss=0.501 lr=5e-05",
        "PROGRESS step=2000/2000 loss=0.433 lr=4.9e-05",
    ]
    assert out[-1] == "STATUS done exit=0"
    assert len(out) == 6

    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "done" and result["exit_code"] == 0
    assert result["last_step"] == 2000 and result["last_loss"] == 0.433
    assert "end_time" in result
    assert "step 1500: eval_loss" in (run / "train.log").read_text(encoding="utf-8")


def test_failing_run_records_failure(tmp_path, capsys):
    rc = main(base_args(tmp_path, write_stub(tmp_path, exit_code=1, lines=(LOG_500,))))
    out = capsys.readouterr().out.splitlines()
    result = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert out[-1] == "STATUS failed exit=1"
    assert result["status"] == "failed" and result["exit_code"] == 1
    assert result["last_step"] == 500


def test_missing_binary_records_failure(tmp_path, capsys):
    rc = main(base_args(tmp_path, str(tmp_path / "no-such-lerobot-train")))
    assert rc == 127
    assert capsys.readouterr().out.splitlines()[-1] == "STATUS failed exit=127"
    assert json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery; lab's runner stops jobs with SIGTERM")
def test_sigterm_forwards_and_records_stopped(tmp_path):
    bin_ = write_stub(tmp_path, lines=[LOG_500] * 200, sleep_s=0.1)
    proc = subprocess.Popen([sys.executable, "-m", "research.train", *base_args(tmp_path, bin_)],
                            cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True)
    seen = []
    for line in proc.stdout:
        seen.append(line.strip())
        if line.startswith("PROGRESS"):
            break
    proc.send_signal(signal.SIGTERM)
    rest, _ = proc.communicate(timeout=30)
    last = rest.strip().splitlines()[-1]

    assert last == "STATUS stopped exit=143"
    assert proc.returncode == 143
    result = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stopped" and result["exit_code"] == 143
