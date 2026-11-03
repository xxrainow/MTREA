import json
import time

import pytest
from fastapi.testclient import TestClient

from lab.core import dataset_edit as edit, dataset_manifest, paths
from lab.core.datasets import get_dataset, list_datasets
from lab.core.hardware import HardwareBusy
from lab.core.hardware.lock import HardwareLock
from lab.server.main import create_app


def source(base, backend='fake', version='v2.1'):
    root = base / ('fake/collection/raw' if backend == 'fake' else 'raw') / 'original'
    meta = root / 'meta'
    meta.mkdir(parents=True)
    info = {'total_episodes': 2, 'total_frames': 4, 'codebase_version': version, 'fake': backend == 'fake'}
    (meta / 'info.json').write_text(json.dumps(info))
    tasks = [{'task_index': 0, 'task': 'Pick red.'}, {'task_index': 1, 'task': 'Pick blue.'}]
    episodes = [{'episode_index': 0, 'tasks': ['Pick red.'], 'length': 2},
                {'episode_index': 1, 'tasks': ['Pick blue.'], 'length': 2}]
    if version == 'v3.0':
        pd = pytest.importorskip('pandas')
        pytest.importorskip('pyarrow')
        pd.DataFrame(tasks).set_index('task').to_parquet(meta / 'tasks.parquet')
        (meta / 'episodes/chunk-000').mkdir(parents=True)
        pd.DataFrame(episodes).to_parquet(meta / 'episodes/chunk-000/file-000.parquet')
    else:
        (meta / 'tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in tasks))
        (meta / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in episodes))
    (root / 'videos').mkdir()
    (root / 'videos/unchanged.mp4').write_bytes(b'synthetic-video-bytes')
    return root


MAPPING = {'Pick red.': '빨간 블록을 옮겨 주세요.', 'Pick blue.': 'Place blue in tray.'}


def fingerprint(root):
    return dataset_manifest.fingerprint(dataset_manifest.manifest(root))


def test_fake_copy_preserves_original_and_collection_history(tmp_path):
    root = source(tmp_path)
    ledger = root.parent / 'original.collection.jsonl'
    ledger.write_text('{"status":"ok"}\n')
    before = fingerprint(root)
    checked = edit.check_edit('fake', 'original', 'corrected', MAPPING, tmp_path)
    assert not (root.parent / 'corrected').exists()
    edit.edit_dataset(checked, tmp_path)
    output = root.parent / 'corrected'
    result = get_dataset('fake', 'corrected', tmp_path)
    assert result.dataset.issues == []
    assert result.episodes[0].tasks == [MAPPING['Pick red.']]
    assert result.episodes[1].tasks == [MAPPING['Pick blue.']]
    assert fingerprint(root) == before
    assert ledger.read_text() == '{"status":"ok"}\n'
    assert not (root.parent / 'corrected.collection.jsonl').exists()
    assert (output / 'videos/unchanged.mp4').read_bytes() == (root / 'videos/unchanged.mp4').read_bytes()
    history = json.loads((output / 'meta/edits.jsonl').read_text())
    assert history['source_fingerprint'] == before
    assert history['git_commit']


@pytest.mark.parametrize('version', ['v2.1', 'v3.0'])
@pytest.mark.parametrize('inline_text', [False, True])
def test_real_task_text_changes_without_numeric_or_video_changes(tmp_path, version, inline_text):
    pd = pytest.importorskip('pandas')
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    root = source(tmp_path, 'real', version)
    data = root / 'data/chunk-000/file-000.parquet'
    data.parent.mkdir(parents=True)
    values = {'task_index': [0, 0, 1, 1], 'episode_index': [0, 0, 1, 1],
              'observation.state': [[1., 2.]] * 4, 'action': [[3., 4.]] * 4}
    if inline_text:
        values['task'] = ['Pick red.', 'Pick red.', 'Pick blue.', 'Pick blue.']
    table = pa.table(values).replace_schema_metadata({b'huggingface': b'{"info":{}}'})
    pq.write_table(table, data, row_group_size=2)
    data_bytes = data.read_bytes()
    before = fingerprint(root)
    checked = edit.check_edit('real', 'original', 'corrected', MAPPING, tmp_path)
    edit.edit_dataset(checked, tmp_path)
    output = root.parent / 'corrected'
    result = get_dataset('real', 'corrected', tmp_path)
    assert result.dataset.issues == []
    assert result.episodes[0].tasks == [MAPPING['Pick red.']]
    new_data = pq.read_table(output / data.relative_to(root))
    for key in ('task_index', 'episode_index', 'observation.state', 'action'):
        assert new_data[key] == table[key]
    assert new_data.schema.metadata == table.schema.metadata
    if inline_text:
        assert new_data['task'].to_pylist() == [MAPPING['Pick red.']] * 2 + [MAPPING['Pick blue.']] * 2
    else:
        assert (output / data.relative_to(root)).read_bytes() == data_bytes
    assert fingerprint(root) == before
    assert (output / 'videos/unchanged.mp4').read_bytes() == (root / 'videos/unchanged.mp4').read_bytes()
    if version == 'v3.0':
        task_table = pd.read_parquet(output / 'meta/tasks.parquet')
        assert task_table.loc[MAPPING['Pick red.'], 'task_index'] == 0
        assert task_table.loc[MAPPING['Pick blue.'], 'task_index'] == 1
    else:
        task_table = [json.loads(row) for row in (output / 'meta/tasks.jsonl').read_text().splitlines()]
        assert [row['task_index'] for row in task_table] == [0, 1]


def test_name_only_copy_and_repeated_edit_history(tmp_path):
    root = source(tmp_path)
    same = {task: task for task in MAPPING}
    edit.edit_dataset(edit.check_edit('fake', 'original', 'renamed', same, tmp_path), tmp_path)
    assert (root.parent / 'renamed/meta/tasks.jsonl').read_bytes() == (root / 'meta/tasks.jsonl').read_bytes()
    edit.edit_dataset(edit.check_edit('fake', 'renamed', 'renamed_again', MAPPING, tmp_path), tmp_path)
    history = (root.parent / 'renamed_again/meta/edits.jsonl').read_text().splitlines()
    assert len(history) == 2
    assert [json.loads(row)['source'] for row in history] == ['original', 'renamed']


@pytest.mark.parametrize('name,mapping', [
    ('original', MAPPING), ('../outside', MAPPING), ('valid', {'Pick red.': 'new'}),
    ('valid', {'Pick red.': ' ', 'Pick blue.': 'new'}),
    ('valid', {'Pick red.': 'same', 'Pick blue.': 'same'}),
])
def test_invalid_edits_are_rejected_without_writes(tmp_path, name, mapping):
    root = source(tmp_path)
    before = fingerprint(root)
    with pytest.raises(ValueError):
        edit.check_edit('fake', 'original', name, mapping, tmp_path)
    assert fingerprint(root) == before
    assert [row.name for row in list_datasets(tmp_path)] == ['original']


def test_source_change_after_review_and_lock(tmp_path):
    root = source(tmp_path)
    checked = edit.check_edit('fake', 'original', 'new', MAPPING, tmp_path)
    with HardwareLock(paths.hardware_lock(tmp_path)) as lock:
        lock.acquire('recording')
        with pytest.raises(HardwareBusy):
            edit.edit_dataset(checked, tmp_path)
    (root / 'extra').write_text('changed')
    with pytest.raises(ValueError, match='changed after review'):
        edit.edit_dataset(checked, tmp_path)
    assert not (root.parent / 'new').exists()


def test_failed_edit_never_publishes_partial_output(tmp_path, monkeypatch):
    root = source(tmp_path)
    before = fingerprint(root)
    checked = edit.check_edit('fake', 'original', 'new', MAPPING, tmp_path)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(edit, 'rewrite_instructions', fail)
    with pytest.raises(OSError, match='disk full'):
        edit.edit_dataset(checked, tmp_path)
    assert fingerprint(root) == before
    assert not (root.parent / 'new').exists()
    assert not list((root.parent / '.editing').iterdir())
    assert HardwareLock(paths.hardware_lock(tmp_path)).holder() is None


def test_symlink_and_prepared_name_are_refused(tmp_path):
    root = source(tmp_path)
    (tmp_path / 'datasets/occupied').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        edit.check_edit('fake', 'original', 'occupied', MAPPING, tmp_path)
    (root / 'external').symlink_to(tmp_path)
    with pytest.raises(dataset_manifest.ManifestError):
        edit.check_edit('fake', 'original', 'new', MAPPING, tmp_path)


def test_api_submits_job_and_opens_edited_copy(tmp_path, monkeypatch):
    source(tmp_path)
    monkeypatch.setenv('MTREA_HARDWARE', 'fake')
    with TestClient(create_app(data_base=tmp_path)) as client:
        url = '/api/datasets/fake/original/edit'
        body = {'name': 'corrected', 'instructions': MAPPING}
        assert client.post(url, json=body).status_code == 422
        response = client.post(url + '/check', json=body)
        assert response.status_code == 200, response.text
        started = client.post(url, json=response.json())
        assert started.status_code == 202, started.text
        job_id = started.json()['job']['id']
        for _ in range(100):
            job = client.get('/api/jobs/' + job_id).json()
            if job['status'] != 'running':
                break
            time.sleep(.05)
        assert job['status'] == 'done', client.get(f'/api/jobs/{job_id}/logs').text
        result = client.get('/api/datasets/fake/corrected').json()
        assert result['episodes'][0]['tasks'] == [MAPPING['Pick red.']]
        assert client.get('/api/datasets/fake/original').json()['episodes'][0]['tasks'] == ['Pick red.']
