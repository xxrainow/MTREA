"""Policy serving management tests without GPU or robot hardware."""

from lab.core.models import Job, JobKind, JobSpec, JobStatus
from lab.core.serving import session_state
from lab.core import serving
from lab.core.remote import RemoteError


def test_stopped_server_does_not_keep_stale_ready_state():
    job = Job(
        id="serve-test",
        spec=JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd="not executed",
        ),
        status=JobStatus.STOPPED,
    )

    old_status = {"status": "ready"}

    assert session_state(job, old_status) == "stopped"

def test_running_server_can_be_ready():
    job = Job(
        id="serve-running",
        spec=JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd="not executed",
        ),
        status=JobStatus.RUNNING,
    )

    assert session_state(job, {"status": "ready"}) == "ready"


def test_unknown_exit_does_not_keep_stale_ready_state():
    job = Job(
        id="serve-unknown",
        spec=JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd="not executed",
        ),
        status=JobStatus.UNKNOWN,
    )

    assert session_state(job, {"status": "ready"}) == "ended_unknown"


def test_connection_error_is_reported_without_marking_job_failed(monkeypatch):
    job = Job(
        id="serve-unreachable",
        spec=JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd="not executed",
        ),
        status=JobStatus.RUNNING,
    )

    def fail_to_read(job, remote):
        raise RemoteError("SSH connection timed out")

    monkeypatch.setattr(serving, "read_session_files", fail_to_read)

    result = serving.session_status(job, [job], None)

    assert result["unreachable"] == "SSH connection timed out"
    assert result["read_error"] is None
    assert result["job"]["status"] == "running"


def test_file_error_is_reported_separately_from_connection_error(monkeypatch):
    job = Job(
        id="serve-file-error",
        spec=JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd="not executed",
        ),
        status=JobStatus.RUNNING,
    )

    def fail_to_read(job, remote):
        raise ValueError("Invalid session JSON")

    monkeypatch.setattr(serving, "read_session_files", fail_to_read)

    result = serving.session_status(job, [job], None)

    assert result["read_error"] == "Invalid session JSON"
    assert result["unreachable"] is None
    assert result["job"]["status"] == "running"

