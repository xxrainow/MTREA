"""Tests for the job-facing side of research.train: --describe, --check, --fake,
the live run files and checkpoint pruning. The fake backend stands in for
lerobot-train; no GPU, no lerobot."""

import json
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import research.train as train
from research.train import (INTERNAL_OPTIONS, METHODS, PARAM_UI, build_parser, main, parse_metrics_line,
                            prune_checkpoints, read_last_step)

REPO_ROOT = Path(__file__).resolve().parent.parent

QUANTILE_STATS = {f: {"mean": [0.0], "std": [1.0], "q01": [-1.0], "q99": [1.0]}
                  for f in ("observation.state", "action")}


def write_dataset(tmp_path: Path, stats=QUANTILE_STATS) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    if stats is not None:
        (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return root


def run_args(tmp_path: Path, **over) -> list[str]:
    if not (tmp_path / "dataset").exists():
        write_dataset(tmp_path)
    opts = {"method": "h0", "dataset": "local/pick_place", "dataset-root": str(tmp_path / "dataset"),
            "output": str(tmp_path / "run"), "seed": "0", "steps": "50", "warmup-steps": "5",
            "log-freq": "10", "save-freq": "20", **over}
    return [f"--{k}={v}" for k, v in opts.items() if v is not None]


def fake_args(tmp_path: Path, **over) -> list[str]:
    return run_args(tmp_path, **{"fake-seconds-per-step": "0", **over}) + ["--fake"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check(tmp_path, capsys, args) -> tuple[int, dict]:
    rc = main(args + ["--check"])
    report = json.loads(capsys.readouterr().out)
    return rc, report


def statuses(report: dict) -> dict[str, str]:
    return {c["id"]: c["status"] for c in report["checks"]}


# --- --describe ---

def test_describe_needs_no_other_flag(capsys):
    assert main(["--describe"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["schema_version"] == 1
    assert d["entrypoint"] == "scripts/train.sh"
    assert d["lerobot_version_tested"] == "0.6.1"
    assert d["run_name_template"] == "{method}_{dataset_name}_s{seed}_{yyyymmdd-HHMM}"
    assert {"config.json", "status.json", "metrics.jsonl", "checkpoints.json", "result.json",
            "train.log"} <= d["run_files"].keys()
    assert d["fixed"]["lerobot_flags"]["wandb.disable_artifact"] == "true"

    methods = {m["name"]: m for m in d["methods"]}
    assert methods["h0"]["max_batch_size_tested"] == 16 and methods["h0"]["oom_batch_size"] == 32
    assert methods["h0"]["seconds_per_step"] == {"1": 0.66, "8": 2.17, "16": 3.92}
    assert methods["expert_only"]["max_batch_size_tested"] == 8
    assert methods["expert_only"]["oom_batch_size"] is None
    assert methods["expert_only"]["seconds_per_step"] == {"1": 0.31, "2": 0.48, "4": 0.83, "8": 1.55}
    assert methods["h0"]["flags"] == METHODS["h0"].flags and methods["h0"]["params"] == []

    params = {p["name"]: p for p in d["params"]}
    by_group = {}
    for p in d["params"]:
        by_group.setdefault(p["group"], []).append(p["name"])
    assert by_group == {
        "basic": ["method", "dataset", "dataset_root", "seed", "output", "steps", "batch_size", "save_freq"],
        "advanced": ["lr", "warmup_steps", "log_freq", "keep_checkpoints", "prune_checkpoints", "wandb",
                     "wandb_project"],
        "fixed": ["base"],
        "fake": ["fake", "fake_seconds_per_step", "fake_fail"],
    }
    assert params["method"]["choices"] == ["expert_only", "h0"]
    assert [n for n, p in params.items() if p["required"]] == ["method", "dataset", "seed", "output"]
    assert params["seed"]["type"] == "int" and params["seed"]["default"] is None
    assert params["steps"] == params["steps"] | {"type": "int", "min": 1, "default": 10000, "flag": "--steps"}
    assert params["lr"]["type"] == "float" and params["lr"]["default"] == 2.5e-5
    assert params["keep_checkpoints"]["type"] == "int_list" and params["keep_checkpoints"]["default"] == []
    assert params["prune_checkpoints"]["type"] == "bool" and params["prune_checkpoints"]["default"] is True
    assert params["wandb_project"]["default"] == "mtrea"
    assert params["fake"]["type"] == "flag"
    assert params["fake_fail"]["choices"] == ["none", "oom", "error"]
    assert all(p["label"] and p["help"] for p in d["params"])


def test_every_cli_option_has_ui_metadata():
    options = {a.dest for a in build_parser()._actions if a.option_strings}
    assert set(INTERNAL_OPTIONS) <= options
    assert options - set(INTERNAL_OPTIONS) == set(PARAM_UI), "add the option to PARAM_UI (or INTERNAL_OPTIONS)"


# --- --check ---

def test_check_ok_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train, "_lerobot_version", lambda: "0.6.1")
    rc, report = check(tmp_path, capsys, run_args(tmp_path, steps="2000", **{"lerobot-bin": sys.executable}))
    assert rc == 0 and report["ok"] is True
    assert statuses(report) == {"output": "ok", "warmup": "ok", "stats": "ok", "lerobot": "ok",
                                "batch_size": "ok", "keep_checkpoints": "ok"}
    assert report["estimated_seconds"] == round(3.92 * 2000)
    assert [p.name for p in tmp_path.iterdir()] == ["dataset"]

    # The command is a complete, explicit run line that parses back to the same options.
    cmd = shlex.split(report["command"])
    assert cmd[0] == "scripts/train.sh" and "--check" not in cmd
    for flag in ("--method=h0", "--seed=0", "--steps=2000", "--batch-size=16", "--save-freq=20",
                 f"--output={tmp_path / 'run'}", "--prune-checkpoints=true", "--wandb=false"):
        assert flag in cmd
    reparsed = vars(build_parser().parse_args(cmd[1:]))
    original = vars(build_parser().parse_args(run_args(tmp_path, steps="2000",
                                                       **{"lerobot-bin": sys.executable})))
    assert reparsed == original


def test_check_fail(tmp_path, capsys):
    write_dataset(tmp_path, stats={f: {"mean": [0.0]} for f in ("observation.state", "action")})
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "result.json").write_text("{}", encoding="utf-8")
    rc, report = check(tmp_path, capsys, run_args(
        tmp_path, steps="5", **{"lerobot-bin": str(tmp_path / "no-such-lerobot-train")}))
    assert rc == 2 and report["ok"] is False
    s = statuses(report)
    assert s["output"] == s["warmup"] == s["stats"] == s["lerobot"] == "fail"
    stats_msg = next(c["message"] for c in report["checks"] if c["id"] == "stats")
    assert "q01/q99" in stats_msg and "--operation.type=recompute_stats" in stats_msg
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["result.json"]


def test_check_warn(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train, "_lerobot_version", lambda: "0.7.0")
    rc, report = check(tmp_path, capsys, run_args(
        tmp_path, **{"batch-size": "32", "keep-checkpoints": "20,30,50", "lerobot-bin": sys.executable}))
    assert rc == 0 and report["ok"] is True
    s = statuses(report)
    assert s["batch_size"] == s["keep_checkpoints"] == s["lerobot"] == "warn"
    msgs = {c["id"]: c["message"] for c in report["checks"]}
    assert "32 ran out of memory" in msgs["batch_size"]
    assert "[30]" in msgs["keep_checkpoints"]  # 20 is a save_freq step, 50 the final step
    assert report["estimated_seconds"] is None  # bs 32 was never measured


def test_check_fake(tmp_path, capsys):
    write_dataset(tmp_path, stats=None)  # like lab's Fake datasets: no stats.json
    rc, report = check(tmp_path, capsys, run_args(
        tmp_path, steps="400", **{"fake-seconds-per-step": "0.5"}) + ["--fake"])
    assert rc == 0
    s = statuses(report)
    assert s["stats"] == "warn" and s["lerobot"] == "ok"
    assert report["estimated_seconds"] == 200.0
    cmd = shlex.split(report["command"])
    assert "--fake" in cmd and "--fake-seconds-per-step=0.5" in cmd and "--fake-fail=none" in cmd
    assert not (tmp_path / "run").exists()


def test_real_check_command_omits_fake_options(tmp_path, capsys):
    _, report = check(tmp_path, capsys, run_args(tmp_path))
    assert "--fake" not in report["command"]


# --- fake end to end ---

def test_fake_run_end_to_end(tmp_path, capsys):
    rc = main(fake_args(tmp_path, **{"keep-checkpoints": "20"}))
    out = capsys.readouterr().out.splitlines()
    run = tmp_path / "run"
    ckpts = run / "lerobot" / "checkpoints"
    assert rc == 0

    # Layout: 40 pruned, 20 kept without training state, the final one intact, `last` -> final.
    assert sorted(p.name for p in ckpts.iterdir()) == ["000020", "000050", "last"]
    assert os.readlink(ckpts / "last") == "000050"
    assert (ckpts / "000020" / "pretrained_model" / "model.safetensors").exists()
    assert not (ckpts / "000020" / "training_state").exists()
    assert (ckpts / "000050" / "training_state" / "training_step.json").exists()

    assert out[0] == "STATUS started"
    assert [l for l in out if l.startswith("PROGRESS")][-1].startswith("PROGRESS step=50/50 ")
    assert [l for l in out if l.startswith("CHECKPOINT")] == [
        "CHECKPOINT step=20", "CHECKPOINT step=40", "CHECKPOINT step=50"]
    assert out[-1] == "STATUS done exit=0"

    status = read_json(run / "status.json")
    assert status["status"] == "done" and status["fake"] is True
    assert (status["method"], status["seed"], status["step"], status["total_steps"]) == ("h0", 0, 50, 50)
    assert status["last_checkpoint_step"] == 50 and status["eta_s"] == 0
    assert status["seconds_per_step"] is not None and status["failure"] is None
    for key in ("loss", "lr", "wandb_url", "started_at", "updated_at"):
        assert key in status

    metrics = read_jsonl(run / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [10, 20, 30, 40, 50]
    for key in ("loss", "lr", "grad_norm", "update_s", "data_s", "samples_per_s", "mem_gb", "epoch"):
        assert key in metrics[0], key
    # Cosine schedule: after warmup the lr only falls, toward the floor (lr / 10). Each
    # logged lr is the window average, so the last one sits just above the floor.
    lrs = [m["lr"] for m in metrics]
    assert lrs == sorted(lrs, reverse=True) and 2.5e-6 <= lrs[-1] < 4e-6

    listing = read_json(run / "checkpoints.json")["checkpoints"]
    assert [(c["step"], c["path"], c["complete"], c["has_training_state"], c["last"]) for c in listing] == [
        (20, "lerobot/checkpoints/000020", True, False, False),
        (50, "lerobot/checkpoints/000050", True, True, True),
    ]
    assert all(c["saved_at"] for c in listing)

    result = read_json(run / "result.json")
    assert result["status"] == "done" and result["exit_code"] == 0
    assert result["last_step"] == 50 and result["last_checkpoint_step"] == 50 and result["fake"] is True

    config = read_json(run / "config.json")
    assert config["fake"] is True and config["lerobot_version"] == "fake"
    assert config["lerobot_argv"][1].endswith("fake_lerobot_train.py")
    assert "Checkpoint policy after step 40" in (run / "train.log").read_text(encoding="utf-8")


def test_fake_run_without_pruning_keeps_everything(tmp_path, capsys):
    assert main(fake_args(tmp_path, **{"prune-checkpoints": "false"})) == 0
    ckpts = tmp_path / "run" / "lerobot" / "checkpoints"
    assert sorted(p.name for p in ckpts.iterdir()) == ["000020", "000040", "000050", "last"]
    assert all((ckpts / s / "training_state").is_dir() for s in ("000020", "000040", "000050"))
    listing = read_json(tmp_path / "run" / "checkpoints.json")["checkpoints"]
    assert [c["step"] for c in listing] == [20, 40, 50] and [c["last"] for c in listing] == [False, False, True]


def test_fake_oom(tmp_path, capsys):
    rc = main(fake_args(tmp_path, **{"fake-fail": "oom"}))
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out.splitlines()[-1] == "STATUS failed exit=1"
    assert "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in captured.err
    assert read_json(tmp_path / "run" / "result.json")["failure"] == "oom"
    status = read_json(tmp_path / "run" / "status.json")
    assert status["status"] == "failed" and status["failure"] == "oom"


def test_fake_error_keeps_completed_checkpoint(tmp_path, capsys):
    rc = main(fake_args(tmp_path, **{"fake-fail": "error"}))  # fails at step 26
    assert rc == 1
    result = read_json(tmp_path / "run" / "result.json")
    assert result["failure"] == "error" and result["last_checkpoint_step"] == 20
    status = read_json(tmp_path / "run" / "status.json")
    assert (status["status"], status["failure"], status["step"]) == ("failed", "error", 20)


def test_fake_run_with_missing_stats_warns(tmp_path, capsys):
    write_dataset(tmp_path, stats=None)
    assert main(fake_args(tmp_path)) == 0
    err = capsys.readouterr().err
    assert "warning: cannot read" in err and "stats.json" in err
    assert read_json(tmp_path / "run" / "status.json")["status"] == "done"


def test_real_run_with_missing_stats_still_refused(tmp_path, capsys):
    write_dataset(tmp_path, stats=None)
    assert main(run_args(tmp_path)) == 2
    assert not (tmp_path / "run").exists()


def test_fake_refuses_existing_output_dir(tmp_path):
    (tmp_path / "out").mkdir()
    proc = subprocess.run([sys.executable, str(train.FAKE_LEROBOT), f"--output_dir={tmp_path / 'out'}",
                           "--steps=10"], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "FileExistsError" in proc.stderr and "already exists and resume is False" in proc.stderr


# --- stop ---

def test_sigterm_sets_status_stopped(tmp_path, capsys, monkeypatch):
    say = train._say
    sent = []

    def say_then_stop(line):
        say(line)
        if line.startswith("PROGRESS") and not sent:
            sent.append(line)
            signal.raise_signal(signal.SIGTERM)  # as lab's runner would, mid-run

    monkeypatch.setattr(train, "_say", say_then_stop)
    rc = main(fake_args(tmp_path, steps="100000", **{"fake-seconds-per-step": "0.001"}))
    out = capsys.readouterr().out.splitlines()
    assert sent
    assert rc == (1 if os.name == "nt" else 143)  # Windows terminate() vs a forwarded SIGTERM
    assert out[-1] == f"STATUS stopped exit={rc}"
    status = read_json(tmp_path / "run" / "status.json")
    assert status["status"] == "stopped" and status["step"] >= 10
    assert read_json(tmp_path / "run" / "result.json")["status"] == "stopped"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery; lab's runner stops jobs with SIGTERM")
def test_sigterm_to_process_sets_status_stopped(tmp_path):
    args = fake_args(tmp_path, steps="100000", **{"fake-seconds-per-step": "0.001"})
    proc = subprocess.Popen([sys.executable, "-m", "research.train", *args],
                            cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("PROGRESS"):
            break
    proc.send_signal(signal.SIGTERM)
    rest, _ = proc.communicate(timeout=30)
    assert rest.strip().splitlines()[-1] == "STATUS stopped exit=143"
    assert read_json(tmp_path / "run" / "status.json")["status"] == "stopped"


# --- W&B ---

@pytest.mark.parametrize("wandb,project", [("false", None), ("true", "mt-rea-ablation")])
def test_wandb_flags(tmp_path, capsys, wandb, project):
    over = {"wandb": wandb, **({"wandb-project": project} if project else {})}
    assert main(run_args(tmp_path, **{"lerobot-bin": "lerobot-train"}, **over) + ["--dry-run"]) == 0
    command = capsys.readouterr().out.splitlines()[-1]
    assert f"--wandb.enable={wandb}" in command
    assert f"--wandb.project={project or 'mtrea'}" in command
    assert "--wandb.disable_artifact=true" in command


def test_wandb_url_is_reported(tmp_path, capsys):
    assert main(fake_args(tmp_path, wandb="true")) == 0
    out = capsys.readouterr().out.splitlines()
    url = "https://wandb.invalid/fake/mtrea/runs/fake-0"
    assert out[1] == f"WANDB url={url}"  # termcolor codes stripped
    assert read_json(tmp_path / "run" / "status.json")["wandb_url"] == url


def test_no_wandb_url_without_wandb(tmp_path, capsys):
    assert main(fake_args(tmp_path)) == 0
    assert not any(l.startswith("WANDB") for l in capsys.readouterr().out.splitlines())
    assert read_json(tmp_path / "run" / "status.json")["wandb_url"] is None


def test_metrics_line_with_ansi_codes():
    line = "\x1b[32mINFO\x1b[0m 2026-10-04 12:00:00 ot_train.py:641 step:200 smpl:3K ep:8 epch:0.16 loss:0.512 " \
           "grdn:3.210 lr:2.5e-05 updt_s:3.920 data_s:0.010 smp/s:4 mem_gb:37.13"
    parsed = parse_metrics_line(train.strip_ansi(line))
    assert parsed == {"step": "200", "loss": 0.512, "lr": 2.5e-05, "epoch": 0.16, "grad_norm": 3.21,
                      "update_s": 3.92, "data_s": 0.01, "samples_per_s": 4.0, "mem_gb": 37.13}


def test_metrics_line_without_mem_gb():
    # mem_gb is only logged when CUDA is available.
    parsed = parse_metrics_line("step:200 smpl:200 ep:1 epch:0.01 loss:0.5 grdn:1.0 lr:1.0e-05 "
                                "updt_s:0.100 data_s:0.001 smp/s:10")
    assert "mem_gb" not in parsed and parsed["samples_per_s"] == 10.0


# --- prune safety ---

def make_checkpoints(ckpt_dir: Path, steps, last=None, training_state=True) -> None:
    for s in steps:
        d = ckpt_dir / f"{s:06d}"
        (d / "pretrained_model").mkdir(parents=True)
        if training_state:
            (d / "training_state").mkdir()
    if last is not None:
        (ckpt_dir / "last").symlink_to(f"{last:06d}")


def names(ckpt_dir: Path) -> list[str]:
    return sorted(p.name for p in ckpt_dir.iterdir())


def test_prune_without_last_deletes_nothing(tmp_path):
    make_checkpoints(tmp_path, [1000, 2000, 3000])
    assert prune_checkpoints(tmp_path) is None
    assert names(tmp_path) == ["001000", "002000", "003000"]


def test_prune_single_dir_without_last_does_not_wait(tmp_path):
    make_checkpoints(tmp_path, [1000])  # the first save is still running
    start = time.monotonic()
    assert read_last_step(tmp_path) is None
    assert time.monotonic() - start < 0.04


def test_prune_spares_last_and_newer(tmp_path):
    make_checkpoints(tmp_path, [1000, 2000], last=2000)
    make_checkpoints(tmp_path, [3000], training_state=False)  # a save in progress
    assert prune_checkpoints(tmp_path) == 2000
    assert names(tmp_path) == ["002000", "003000", "last"]
    assert (tmp_path / "002000" / "training_state").is_dir()
    assert os.readlink(tmp_path / "last") == "002000"


def test_prune_keep_drops_only_training_state(tmp_path):
    make_checkpoints(tmp_path, [1000, 2000, 3000], last=3000)
    assert prune_checkpoints(tmp_path, keep=[1000]) == 3000
    assert names(tmp_path) == ["001000", "003000", "last"]
    assert (tmp_path / "001000" / "pretrained_model").is_dir()
    assert not (tmp_path / "001000" / "training_state").exists()
    assert (tmp_path / "003000" / "training_state").is_dir()


def test_last_read_retries_through_a_swap(tmp_path):
    make_checkpoints(tmp_path, [1000, 2000])  # `last` unlinked, not yet re-created
    timer = threading.Timer(0.1, lambda: (tmp_path / "last").symlink_to("002000"))
    timer.start()
    try:
        assert read_last_step(tmp_path, retries=40, delay_s=0.01) == 2000
    finally:
        timer.join()


def test_last_pointing_elsewhere_is_unknown(tmp_path):
    make_checkpoints(tmp_path, [1000, 2000])
    (tmp_path / "last").symlink_to("not-a-step")
    assert prune_checkpoints(tmp_path) is None
    assert names(tmp_path) == ["001000", "002000", "last"]


def test_every_checkpoint_is_announced_when_last_jumps(tmp_path, capsys):
    # The reader lagged: `last` moved 20 -> 50 between two reads.
    run = tmp_path / "run"
    make_checkpoints(run / "lerobot" / "checkpoints", [20, 40, 50], last=50)
    monitor = train._Run(build_parser().parse_args(run_args(tmp_path)), run)
    monitor.status["last_checkpoint_step"] = 20
    monitor.refresh_checkpoints(prune=True)
    assert capsys.readouterr().out.splitlines() == ["CHECKPOINT step=40", "CHECKPOINT step=50"]
    assert names(run / "lerobot" / "checkpoints") == ["000050", "last"]


def test_stub_checkpoint_lines_drive_pruning(tmp_path, capsys):
    """Pruning runs on lerobot's "Checkpoint policy" line, before the next save completes."""
    ckpts = tmp_path / "run" / "lerobot" / "checkpoints"
    stub = tmp_path / "stub.py"
    stub.write_text(textwrap.dedent(f"""
        import os, sys, time
        from pathlib import Path
        ckpts = Path({str(ckpts)!r})
        for step in (20, 40):
            print(f"INFO x ot_train.py:687 Checkpoint policy after step {{step}}", file=sys.stderr, flush=True)
            time.sleep(0.2)
            d = ckpts / f"{{step:06d}}"
            (d / "pretrained_model").mkdir(parents=True)
            (d / "training_state").mkdir()
            if (ckpts / "last").is_symlink():
                (ckpts / "last").unlink()
            (ckpts / "last").symlink_to(d.name)
        # The third save starts but the process dies before `last` moves.
        print("INFO x ot_train.py:687 Checkpoint policy after step 50", file=sys.stderr, flush=True)
        time.sleep(0.2)
        (ckpts / "000050" / "pretrained_model").mkdir(parents=True)
        sys.exit(1)
    """), encoding="utf-8")
    rc = main(run_args(tmp_path, **{"lerobot-bin": f'"{sys.executable}" "{stub}"'}))
    assert rc == 1
    assert names(ckpts) == ["000040", "000050", "last"]  # 50 is newer than last: untouched
    assert read_json(tmp_path / "run" / "result.json")["last_checkpoint_step"] == 40
    listing = read_json(tmp_path / "run" / "checkpoints.json")["checkpoints"]
    assert [(c["step"], c["complete"]) for c in listing] == [(40, True), (50, False)]
