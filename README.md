# MTREA
Multi-Task Retentive Embodiment Adaptation



## Lab API

Run these commands from the repository root in an activated Python environment.

### Install

```bash
python -m pip install -e ".[server,dev]"
```

### Start the development server

```bash
python -m uvicorn lab.server.main:app \
  --host 127.0.0.1 \
  --port 8001 \
  --reload
```

- API docs: http://127.0.0.1:8001/docs
- Health: http://127.0.0.1:8001/api/health

Use a single server worker. The current runner lock is process-local.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Check API availability |
| GET | `/api/jobs` | Refresh and list jobs |
| GET | `/api/jobs/{job_id}` | Get a job |
| GET | `/api/jobs/{job_id}/logs` | Read the last log lines |
| POST | `/api/jobs/{job_id}/stop` | Stop a job |

Job submission through HTTP is not implemented yet.

### Data location

Data defaults to the repository's `data/` directory.
Set `ROBOLAB_DATA_ROOT` to use another location.

Jobs are recorded in `jobs/jobs.jsonl`, and job output is stored in `logs/`.

### Tests

```bash
python -m pytest tests/test_api.py tests/test_boundary.py -v
```

API tests use temporary data directories.
