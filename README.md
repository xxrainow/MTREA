# MTREA

Multi-Task Retentive Embodiment Adaptation

## Local development

Run the backend and frontend in separate terminals.

### Backend installation

From the repository root, in an activated Python environment:

```bash
python -m pip install -e ".[server,dev]"
```

### Start the backend

From the repository root:

```bash
python -m uvicorn lab.server.main:app \
  --host 127.0.0.1 \
  --port 8001 \
  --reload
```

- API docs: http://127.0.0.1:8001/docs
- Health: http://127.0.0.1:8001/api/health

Use a single server worker. The current runner lock is process-local.

### Frontend installation

In a separate terminal, from the repository root:

```bash
cd lab/web
npm install
```

### Start the frontend

From `lab/web/`:

```bash
npm run dev -- --host 127.0.0.1 --port 8080 --strictPort
```

Open http://127.0.0.1:8080 in your browser.

The Vite development server must proxy `/api` requests to
`http://127.0.0.1:8001`, preserving the request path.

For example, a browser request to `/api/jobs` on port 8080 is forwarded
to `/api/jobs` on port 8001. Keep both servers running.

This proxy is for local development. Production hosting requires
a separate configuration.

### Hardware UI

Devices and Teleoperate use the hardware API. On the computer connected to
the arms, open Devices to set each serial port (choose or type it and save,
or use Detect port). Detection snapshots the ports first: unplug only the
selected arm, press Check removed port, plug it back in, then save.

Calibrate each arm from Devices. Follow the status instruction: support the
arm with torque off, place it near the middle of its range and press Next,
then move the joints through their full ranges and press Finish and save.
Cancel exits without saving. The table shows raw encoder readings, not degrees.

Open Teleoperate to start leader-to-follower control. The page shows achieved
FPS and joint positions from `/ws/joints`; Vite proxies `/ws` with WebSocket
support alongside `/api`. Rest the follower before pressing Stop: stopping
disables its torque. Changing pages does not stop calibration or teleoperation;
return to the corresponding page to cancel or stop. Stop teleoperation before
starting CLI collection. Camera previews and Collect are still pending.

For UI development without arms, launch the backend in fake mode:

```bash
MTREA_HARDWARE=fake python -m uvicorn lab.server.main:app --host 127.0.0.1 --port 8001
```

Both pages explicitly label fake hardware. Fake mode simulates port detection
and joint motion and writes separate fake calibration files. It permits teleop
without calibration; real mode requires both calibration files and present
ports before the UI enables Start. A present port or file alone does not prove
that the robot can connect; connection errors appear on the page.

## Lab API

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Check API availability |
| GET | `/api/jobs` | Refresh and list jobs |
| GET | `/api/jobs/{job_id}` | Get a job |
| GET | `/api/jobs/{job_id}/logs` | Read the last log lines |
| POST | `/api/jobs/{job_id}/stop` | Stop a job |
| GET | `/api/hardware/config` | Arm ports, ids, calibration files, who holds the arms |
| GET | `/api/hardware/ports` | List serial ports |
| POST | `/api/hardware/ports/detect/start` | `{arm}`: snapshot the ports before unplugging that arm |
| POST | `/api/hardware/ports/detect/finish` | `{arm}`: wait up to 15 s for one port to disappear |
| POST | `/api/hardware/ports/save` | `{arm, port}`: write the port into `robot/configs/so101_<arm>.yaml` |
| POST | `/api/calibration/start` | `{arm}`: start calibrating one arm |
| POST | `/api/calibration/next` | Finish the current step (homing, then range recording) |
| POST | `/api/calibration/cancel` | Cancel; nothing is saved |
| GET | `/api/calibration/status` | Step, instruction text, per-joint min/max/current |
| POST | `/api/teleop/start` | Open both arms and start leader → follower |
| POST | `/api/teleop/stop` | Stop teleoperation |
| GET | `/api/teleop/status` | Running, achieved fps, error |
| WS | `/ws/joints` | Joint frames during teleoperation and calibration |

Hardware errors carry a `detail` message for the UI: 409 when the arms are
held by another session or the request does not fit the current step, 422
when port detection fails, 503 when an arm cannot be opened.

Job submission through HTTP is not implemented yet.

### Job status limitations

A job's exit code may be unavailable when it was started by another
process or its Popen handle was lost after a server restart.

When the exit code is unavailable, the runner leaves the ledger
unchanged rather than marking the job as failed. The process that
owns the Popen handle must refresh and persist the final result.

If that process exits before saving the result, a completed job
may remain recorded as `running`. Durable exit-result storage and
explicit handling of unknown outcomes are future work.

## Hardware

`lab/core/hardware/` handles port detection, calibration and teleoperation for
the SO-101 arms. The same code runs from the terminal, without the UI:

```bash
python -m lab.core.hardware.cli ports
python -m lab.core.hardware.cli detect --arm=follower --save
python -m lab.core.hardware.cli calibrate --arm=leader
python -m lab.core.hardware.cli teleop --seconds=30
```

- **Settings.** Ports, ids and `use_degrees` come from
  `robot/configs/so101_*.yaml`; saving a detected port rewrites only the
  `port:` line. Calibration files are written where lerobot v0.6.1 reads them:
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json` and
  `.../teleoperators/so_leader/<id>.json`. The `so101_follower/` and
  `so101_leader/` directories used by older lerobot versions are not read.
- **One holder.** Calibration, teleoperation and `experiments/collect.py`
  take one lock (`data/locks/hardware.lock`, holding the owner's pid). A
  second one is refused until the first stops. A lock left by a process that
  died is reclaimed. A bare `lerobot-record` run does not take the lock.
- **Stopping teleoperation turns the follower's torque off** (lerobot's
  `disable_torque_on_disconnect`), so the arm goes limp. Rest it first.
- **Fake mode.** `MTREA_HARDWARE=fake` swaps in fake arms (ports, calibration,
  smoothly moving joints) for development without hardware. It is never the
  default. Fake mode edits copies of the arm yamls under `data/fake/` and
  saves calibrations as `<id>.fake.json`, which lerobot never reads. Every
  status reports `backend: "real" | "fake"`.

## Data location

Data defaults to the repository's `data/` directory.
Set `ROBOLAB_DATA_ROOT` to use another location.

Jobs are recorded in `jobs/jobs.jsonl`, and job output is stored in
`logs/`, relative to the data root.

## Tests

From the repository root:

```bash
python -m pytest tests/test_api.py tests/test_boundary.py -v
python -m pytest tests/test_hardware.py tests/test_hardware_api.py -v
```

The hardware tests use fake arms and need neither lerobot nor a robot.

API tests use temporary data directories.
