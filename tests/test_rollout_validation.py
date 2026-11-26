"""Repeated validation with fake job execution; no devices, GPU, or network."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import psutil
import pytest
from fastapi.testclient import TestClient

from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.rollout_validation import RolloutValidationService
from lab.server.main import create_app


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def client(tmp_path):
    app = create_app(data_base=tmp_path / 'data', remote_config=tmp_path / 'missing.yaml')
    with TestClient(app) as client:
        executor = Mock()
        executor.start.return_value = 12345
        executor.is_running.return_value = True
        app.state.runner.executors[Executor.LOCAL] = executor
        app.state.cameras.stop = Mock()
        root = tmp_path / 'server'
        write(root / 'status.json', {'status': 'ready', 'port': 19234})
        write(root / 'session.json', {'checkpoint': 'test checkpoint'})
        app.state.runner.store.save(Job(id='server', spec=JobSpec(
            kind=JobKind.SERVE_POLICY, cmd='never run', run_dir=str(root),
            metadata={'checkpoint': 'test checkpoint', 'server_port': 19234}),
            pid=45678, status=JobStatus.RUNNING))
        yield client


def settings(**changes):
    return {'server_job_id': 'server', 'instruction': 'Pick / blue "block"; $(no-shell)',
            'max_relative_target': 5.0, 'time_limit_s': 30.0, 'target_count': 2, **changes}


def start(client, **changes):
    body = settings(request_id=uuid4().hex, **changes)
    result = client.post('/api/rollout-validations/start', json=body)
    assert result.status_code == 201, result.text
    return result.json(), body


def next_attempt(client, value, **changes):
    return client.post(f"/api/rollout-validations/{value['id']}/next", json={
        'request_id': uuid4().hex, 'number': value['started_count'] + 1, 'scene_ready': True, **changes})


def trial_status(client, value, *, alive=False, state='stopped', released=False, running_job=False):
    job_id = value['attempts'][-1]['rollout']['job']['id']
    job = client.app.state.runner.store.get(job_id)
    job.status = JobStatus.RUNNING if running_job else JobStatus.DONE
    client.app.state.runner.store.save(job)
    write(Path(job.spec.run_dir) / 'status.json', {
        'state': state, 'process_alive': alive, 'pid': os.getpid(),
        'process_started': psutil.Process().create_time(), 'arm_released': released,
        'workers_stopped': state != 'running', 'elapsed_s': 2.0,
    })
    return job


def read(client, value):
    return next(item for item in client.get('/api/rollout-validations').json() if item['id'] == value['id'])


def test_check_writes_nothing_and_attempt_count_is_strict(client):
    root = client.app.state.rollout_validation.root
    assert client.post('/api/rollout-validations/check', json=settings()).status_code == 200
    assert not root.exists()
    for count in [0, -1, 1001, True, 1.5, '2']:
        assert client.post('/api/rollout-validations/check', json=settings(target_count=count)).status_code == 422
    client.app.state.runner.executors[Executor.LOCAL].start.assert_not_called()
    client.app.state.cameras.stop.assert_not_called()


def test_repeat_requires_release_and_never_starts_on_reads(client):
    value, body = start(client)
    executor = client.app.state.runner.executors[Executor.LOCAL]
    assert value['started_count'] == 1 and not value['can_next']
    assert next_attempt(client, value).status_code == 409
    trial_status(client, value, alive=True, released=False, running_job=True)
    assert next_attempt(client, value).status_code == 409
    trial_status(client, value, released=False)
    assert next_attempt(client, value).status_code == 409
    trial_status(client, value, released=True, running_job=True)
    assert next_attempt(client, value).status_code == 409
    trial_status(client, value, released=True)
    ready = read(client, value)
    assert ready['state'] == 'awaiting_next' and ready['can_next']
    for _ in range(3):
        read(client, value)
    executor.start.assert_called_once()
    second = next_attempt(client, ready)
    assert second.status_code == 201, second.text
    second = second.json()
    assert [item['number'] for item in second['attempts']] == [1, 2]
    metadata = second['attempts'][-1]['rollout']['job']['spec']['metadata']
    assert metadata['instruction'] == body['instruction']
    assert metadata['checkpoint'] == 'test checkpoint'
    assert metadata['validation']['id'] == value['id']
    trial_status(client, second, released=True)
    completed = read(client, value)
    assert completed['state'] == 'completed' and not completed['can_next']
    assert completed['released_count'] == 2 and completed['judged_count'] == 0
    assert next_attempt(client, completed).status_code == 409
    assert executor.start.call_count == 2


def test_restart_recovers_and_duplicate_requests_do_not_submit(client):
    value, body = start(client)
    original = (client.app.state.rollout_validation.directory(value['id']) / 'session.json').read_bytes()
    client.app.state.rollout_validation = RolloutValidationService(client.app.state.rollout)
    assert read(client, value)['started_count'] == 1
    assert client.post('/api/rollout-validations/start', json=body).status_code == 201
    assert client.post('/api/rollout-validations/start', json={**body, 'target_count': 3}).status_code == 409
    trial_status(client, value, released=True)
    request = {'number': 2, 'request_id': uuid4().hex, 'scene_ready': True}
    url = f"/api/rollout-validations/{value['id']}/next"
    assert client.post(url, json=request).status_code == 201
    assert client.post(url, json=request).status_code == 201
    assert (client.app.state.rollout_validation.directory(value['id']) / 'session.json').read_bytes() == original
    assert client.app.state.runner.executors[Executor.LOCAL].start.call_count == 2


def test_concurrent_next_requests_submit_one_trial_and_reject_stale_number(client):
    value, _ = start(client)
    trial_status(client, value, released=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: next_attempt(client, value).status_code, range(2)))
    assert sorted(results) == [201, 409]
    updated = read(client, value)
    trial_status(client, updated, released=True)
    assert next_attempt(client, updated, number=2).status_code == 409
    assert client.app.state.runner.executors[Executor.LOCAL].start.call_count == 2


def test_annotations_are_append_only_and_errors_do_not_assign_failure(client):
    value, _ = start(client)
    job = trial_status(client, value, state='failed', released=True)
    service = client.app.state.rollout_validation
    events = service.directory(value['id']) / 'events.jsonl'
    before = events.read_bytes()
    assert read(client, value)['attempts'][0]['outcome'] == 'unjudged'
    url = f"/api/rollout-validations/{value['id']}/outcome"
    response = client.post(url, json={'job_id': job.id, 'outcome': 'success', 'notes': 'Reached the target'})
    assert response.status_code == 200
    assert response.json()['judged_count'] == 1
    after = events.read_bytes()
    assert after.startswith(before)
    response = client.post(url, json={'job_id': job.id, 'outcome': 'failure', 'notes': 'Correction'})
    assert response.status_code == 200 and events.read_bytes().startswith(after)
    assert read(client, value)['attempts'][0]['outcome'] == 'failure'
    assert client.post(url, json={'job_id': 'other', 'outcome': 'success'}).status_code == 404


def test_early_end_preserves_partial_results_and_never_stops_hardware(client):
    value, _ = start(client, target_count=10)
    url = f"/api/rollout-validations/{value['id']}/end"
    assert client.post(url).status_code == 409
    trial_status(client, value, released=True)
    result = client.post(url)
    assert result.status_code == 200 and result.json()['state'] == 'ended'
    assert result.json()['started_count'] == 1
    assert result.json()['session']['target_count'] == 10
    assert next_attempt(client, value).status_code == 409
    client.app.state.runner.executors[Executor.LOCAL].stop.assert_not_called()


def test_checkpoint_changes_and_competing_validations_are_refused(client):
    value, _ = start(client)
    assert client.post('/api/rollout-validations/start', json=settings(request_id=uuid4().hex)).status_code == 409
    trial_status(client, value, released=True)
    job = client.app.state.runner.store.get('server')
    job.spec.metadata['checkpoint'] = 'different model'
    client.app.state.runner.store.save(job)
    result = next_attempt(client, value)
    assert result.status_code == 400 and 'checkpoint changed' in result.text
    client.app.state.runner.executors[Executor.LOCAL].start.assert_called_once()


def test_unresolved_start_blocks_retry_and_partial_events_are_not_rewritten(client):
    value, _ = start(client)
    trial_status(client, value, released=True)
    service = client.app.state.rollout_validation
    service.append(value['id'], {'type': 'attempt_requested', 'request_id': uuid4().hex, 'number': 2})
    assert read(client, value)['state'] == 'blocked'
    assert next_attempt(client, value).status_code == 409
    events = service.directory(value['id']) / 'events.jsonl'
    with events.open('ab') as stream:
        stream.write(b'{"type":')
    original = events.read_bytes()
    assert read(client, value)['state'] == 'blocked'
    assert client.post(f"/api/rollout-validations/{value['id']}/end").status_code == 400
    assert events.read_bytes() == original


def test_fake_hardware_and_collection_refuse_validation_start(client):
    client.app.state.hardware.backend = Mock()
    client.app.state.hardware.backend.name = 'fake'
    assert client.post('/api/rollout-validations/start', json=settings(request_id=uuid4().hex)).status_code == 400
    assert not client.app.state.rollout_validation.root.exists()
    client.app.state.hardware.backend.name = 'real'
    client.app.state.collection.is_active = Mock(return_value=True)
    assert client.post('/api/rollout-validations/start', json=settings(request_id=uuid4().hex)).status_code == 409
    client.app.state.runner.executors[Executor.LOCAL].start.assert_not_called()


def test_next_requires_explicit_scene_readiness(client):
    value, _ = start(client)
    trial_status(client, value, released=True)
    assert next_attempt(client, value, scene_ready=False).status_code == 400
    assert next_attempt(client, value, scene_ready=1).status_code == 422
    client.app.state.runner.executors[Executor.LOCAL].start.assert_called_once()


def test_start_failure_is_preserved_and_never_silently_retried(client):
    executor = client.app.state.runner.executors[Executor.LOCAL]
    executor.start.side_effect = OSError('synthetic launch error')
    request_id = uuid4().hex
    response = client.post('/api/rollout-validations/start', json=settings(request_id=request_id))
    assert response.status_code == 503
    value = client.get('/api/rollout-validations').json()[0]
    assert value['state'] == 'blocked' and value['started_count'] == 0
    assert value['start_errors'][0]['error'] == 'synthetic launch error'
    assert client.post('/api/rollout-validations/start', json=settings(request_id=request_id)).status_code == 201
    assert next_attempt(client, value).status_code == 409
    executor.start.assert_called_once()


def test_reads_do_not_recheck_server_and_standalone_start_cannot_bypass_validation(client):
    value, _ = start(client)
    trial_status(client, value, released=True)
    client.app.state.rollout.check = Mock(side_effect=AssertionError('Read must not probe the policy server'))
    assert read(client, value)['can_next']
    body = {key: val for key, val in settings().items() if key != 'target_count'}
    assert client.post('/api/rollouts/start', json=body).status_code == 409
    client.app.state.runner.executors[Executor.LOCAL].start.assert_called_once()


def test_result_archive_is_reversible_and_preserves_all_source_records(client):
    value, _ = start(client, target_count=1)
    job = trial_status(client, value, released=True)
    validation_dir = client.app.state.rollout_validation.directory(value['id'])
    trial_dir = Path(job.spec.run_dir)
    before = {path: path.read_bytes() for root in (validation_dir, trial_dir) for path in root.rglob('*') if path.is_file()}
    body = {'kind': 'validation', 'id': value['id']}
    response = client.post('/api/rollout-results/archive', json=body)
    assert response.status_code == 200, response.text
    assert response.json()['archived'] == [body]
    archive = client.app.state.rollout_result_archive
    archived_bytes = archive.path.read_bytes()
    assert client.post('/api/rollout-results/archive', json=body).status_code == 200
    assert archive.path.read_bytes() == archived_bytes
    assert read(client, value)['started_count'] == 1
    assert any(item['job']['id'] == job.id for item in client.get('/api/rollouts').json())
    from lab.core.rollout_result_archive import RolloutResultArchive
    client.app.state.rollout_result_archive = RolloutResultArchive(client.app.state.rollout_validation)
    assert client.get('/api/rollout-results/archive').json()['archived'] == [body]
    response = client.post('/api/rollout-results/restore', json=body)
    assert response.status_code == 200 and response.json()['archived'] == []
    assert archive.path.read_bytes().startswith(archived_bytes)
    assert all(path.read_bytes() == content for path, content in before.items())
    client.app.state.runner.executors[Executor.LOCAL].stop.assert_not_called()


def test_result_archive_refuses_active_or_unfinished_validation_and_individual_linked_trials(client):
    value, _ = start(client)
    body = {'kind': 'validation', 'id': value['id']}
    assert client.post('/api/rollout-results/archive', json=body).status_code == 409
    trial_status(client, value, alive=True, running_job=False)
    assert client.post('/api/rollout-results/archive', json=body).status_code == 409
    job = trial_status(client, value, released=True)
    assert client.post('/api/rollout-results/archive', json=body).status_code == 409
    assert client.post('/api/rollout-results/archive', json={'kind': 'standalone', 'id': job.id}).status_code == 409
    assert client.post(f"/api/rollout-validations/{value['id']}/end").status_code == 200
    assert client.post('/api/rollout-results/archive', json=body).status_code == 200
    assert not (Path(job.spec.run_dir) / 'commands').exists()
    client.app.state.runner.executors[Executor.LOCAL].stop.assert_not_called()


def test_legacy_result_archive_and_restore_do_not_convert_or_delete_trial(client):
    root = client.app.state.rollout.root / 'legacy-result'
    job = Job(id='legacy-result', spec=JobSpec(kind=JobKind.ROLLOUT, cmd='never run', run_dir=str(root)), status=JobStatus.DONE)
    client.app.state.runner.store.save(job)
    write(root / 'status.json', {'state': 'stopped', 'process_alive': False, 'arm_released': True, 'elapsed_s': 1})
    write(root / 'session.json', {'task': 'Historical instruction', 'checkpoint': 'historical policy'})
    before = {path: path.read_bytes() for path in root.iterdir()}
    body = {'kind': 'standalone', 'id': job.id}
    assert client.post('/api/rollout-results/archive', json=body).status_code == 200
    assert client.get('/api/rollout-results/archive').json()['archived'] == [body]
    assert client.post('/api/rollout-results/restore', json=body).status_code == 200
    assert not client.app.state.rollout_validation.root.exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_result_archive_rejects_invalid_requests_and_torn_appends(client):
    assert client.post('/api/rollout-results/archive', json={'kind': 'standalone', 'id': 'missing'}).status_code == 404
    assert client.post('/api/rollout-results/archive', json={'kind': 'wrong', 'id': 'missing'}).status_code == 422
    assert client.post('/api/rollout-results/archive', json={'kind': 'validation', 'id': '../outside'}).status_code == 400
    value, _ = start(client, target_count=1)
    trial_status(client, value, released=True)
    body = {'kind': 'validation', 'id': value['id']}
    assert client.post('/api/rollout-results/archive', json=body).status_code == 200
    path = client.app.state.rollout_result_archive.path
    with path.open('ab') as stream:
        stream.write(b'{"action":')
    original = path.read_bytes()
    assert client.post('/api/rollout-results/restore', json=body).status_code == 400
    assert path.read_bytes() == original


def finished_validation(client, **changes):
    value, _ = start(client, target_count=1, **changes)
    trial_status(client, value, released=True)
    return read(client, value)


def merge_body(*values):
    return {'sources': [{'kind': 'validation', 'id': value['id']} for value in values],
            'name': 'Combined validation', 'request_id': uuid4().hex}


def test_result_merge_preserves_sources_and_recovers_with_archive(client):
    first = finished_validation(client)
    second = finished_validation(client)
    service = client.app.state.rollout_validation
    source_files = {path: path.read_bytes() for value in (first, second)
                    for path in service.directory(value['id']).iterdir() if path.is_file()}
    job_id = first['attempts'][0]['rollout']['job']['id']
    client.post(f"/api/rollout-validations/{first['id']}/outcome",
                json={'job_id': job_id, 'outcome': 'success', 'notes': 'Saved judgment'})
    source_files = {path: path.read_bytes() for path in source_files}
    body = merge_body(first, second)
    result = client.post('/api/rollout-results/merge', json=body)
    assert result.status_code == 201, result.text
    merged = result.json()
    assert [a['number'] for a in merged['attempts']] == [1, 2]
    assert merged['attempts'][0]['outcome'] == 'success'
    assert merged['attempts'][0]['notes'] == 'Saved judgment'
    assert merged['attempts'][0]['source']['id'] == first['id']
    assert merged['target_count'] == 2
    assert client.post('/api/rollout-results/merge', json=body).json() == merged
    assert client.get('/api/rollout-results/merged').json() == [merged]
    assert all(path.read_bytes() == data for path, data in source_files.items())
    archive = {'kind': 'merged', 'id': merged['id']}
    assert client.post('/api/rollout-results/archive', json=archive).json()['archived'] == [archive]
    assert client.post('/api/rollout-results/restore', json=archive).json()['archived'] == []
    assert client.get('/api/rollout-results/merged').json() == [merged]
    # Later source judgments do not silently change a saved merged report.
    client.post(f"/api/rollout-validations/{first['id']}/outcome",
                json={'job_id': job_id, 'outcome': 'failure', 'notes': 'Later judgment'})
    assert client.get('/api/rollout-results/merged').json()[0]['attempts'][0]['outcome'] == 'success'
    overlap = {**merge_body(first), 'sources': [archive, {'kind': 'validation', 'id': first['id']}]}
    refused = client.post('/api/rollout-results/merge', json=overlap)
    assert refused.status_code == 409 and 'overlap' in refused.text


def test_result_merge_refuses_different_conditions_active_and_deleted(client):
    first = finished_validation(client)
    second = finished_validation(client, instruction='Different instruction')
    assert client.post('/api/rollout-results/merge', json=merge_body(first, second)).status_code == 409
    third, _ = start(client, target_count=1)
    assert client.post('/api/rollout-results/merge', json=merge_body(first, third)).status_code == 409
    trial_status(client, third, released=True)
    ref = {'kind': 'validation', 'id': first['id']}
    assert client.post('/api/rollout-results/archive', json=ref).status_code == 200
    assert client.post('/api/rollout-results/merge', json=merge_body(first, third)).status_code == 409
    assert client.get('/api/rollout-results/merged').json() == []
