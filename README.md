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

## Lab API

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Check API availability |
| GET | `/api/jobs` | Refresh and list jobs |
| GET | `/api/jobs/{job_id}` | Get a job |
| GET | `/api/jobs/{job_id}/logs` | Read the last log lines |
| POST | `/api/jobs/{job_id}/stop` | Stop a job |

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

## Data location

Data defaults to the repository's `data/` directory.
Set `ROBOLAB_DATA_ROOT` to use another location.

Jobs are recorded in `jobs/jobs.jsonl`, and job output is stored in
`logs/`, relative to the data root.

## Tests

From the repository root:

```bash
python -m pytest tests/test_api.py tests/test_boundary.py -v
```

API tests use temporary data directories.
