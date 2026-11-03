import json
import time

import pytest
from fastapi.testclient import TestClient

from lab.core import dataset_merge as merge
from lab.core.dataset_manifest import ManifestError, fingerprint, manifest
from lab.core.datasets import get_dataset, list_datasets
from lab.core.dataset_delete import archive_dataset
from lab.core.hardware import HardwareBusy
from lab.core.hardware.lock import HardwareLock
from lab.core import paths
from lab.server.main import create_app


def fake(base, name, count=2, task='Move the block.'):
    root = base / 'fake/collection/raw' / name
    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({'fake': True, 'total_episodes': count}))
    (meta / 'tasks.jsonl').write_text(json.dumps({'task': task}) + '\n')
    (meta / 'episodes.jsonl').write_text(''.join(json.dumps({'episode_index': i, 'tasks': [task]}) + '\n' for i in range(count)))
    return root


@pytest.fixture
def sources(tmp_path):
    return [fake(tmp_path, 'one'), fake(tmp_path, 'two', 3, 'Place the block.')]


def test_fake_merge_preserves_sources_and_records_provenance(tmp_path, sources):
    before = [fingerprint(manifest(root)) for root in sources]
    checked = merge.check_merge('fake', ['one', 'two'], 'combined', tmp_path)
    assert checked['episodes'] == 5
    assert checked['total_frames'] is None
    assert not (tmp_path / checked['location']).exists()
    merge.merge_datasets('fake', ['one', 'two'], 'combined', tmp_path, checked['source_versions'])
    assert [fingerprint(manifest(root)) for root in sources] == before
    result = get_dataset('fake', 'combined', tmp_path)
    assert result.dataset.issues == []
    assert [ep.index for ep in result.episodes] == list(range(5))
    assert result.episodes[2].tasks == ['Place the block.']
    provenance = json.loads((tmp_path / checked['location'] / 'meta/merge.json').read_text())
    assert [row['fingerprint'] for row in provenance['sources']] == before
    assert [row['output_episode_start'] for row in provenance['sources']] == [0, 2]
    assert provenance['git_commit']
    archive_dataset('fake', 'combined', tmp_path)
    assert {row.name for row in list_datasets(tmp_path)} == {'one', 'two'}


@pytest.mark.parametrize('names,name', [(['one'], 'new'), (['one', 'one'], 'new'), (['one', 'two'], 'one'), (['one', 'two'], '../outside'), (['one', 'two'], '.hidden')])
def test_rejects_invalid_selections(tmp_path, sources, names, name):
    with pytest.raises(ValueError):
        merge.check_merge('fake', names, name, tmp_path)


def test_rejects_changed_review_and_symlink(tmp_path, sources):
    checked = merge.check_merge('fake', ['one', 'two'], 'new', tmp_path)
    (sources[0] / 'changed').write_text('new data')
    with pytest.raises(ValueError, match='changed after review'):
        merge.merge_datasets('fake', ['one', 'two'], 'new', tmp_path, checked['source_versions'])
    (sources[0] / 'link').symlink_to(tmp_path)
    with pytest.raises(ManifestError):
        merge.check_merge('fake', ['one', 'two'], 'new', tmp_path)


def test_failure_does_not_publish_partial_output(tmp_path, sources, monkeypatch):
    def fail(roots, output):
        (output / 'meta').mkdir(parents=True)
        (output / 'meta/info.json').write_text('{}')
        raise RuntimeError('disk full')
    monkeypatch.setattr(merge, '_merge_fake', fail)
    with pytest.raises(RuntimeError, match='disk full'):
        merge.merge_datasets('fake', ['one', 'two'], 'new', tmp_path)
    assert {row.name for row in list_datasets(tmp_path)} == {'one', 'two'}
    assert not list((sources[0].parent / '.merging').iterdir())
    assert HardwareLock(paths.hardware_lock(tmp_path)).holder() is None


def test_source_mutation_during_merge_not_published(tmp_path, sources, monkeypatch):
    original = merge._merge_fake
    def change(roots, output):
        original(roots, output)
        (roots[0] / 'changed').write_text('changed')
    monkeypatch.setattr(merge, '_merge_fake', change)
    with pytest.raises(ValueError, match='Source data changed'):
        merge.merge_datasets('fake', ['one', 'two'], 'new', tmp_path)
    assert not (sources[0].parent / 'new').exists()


def test_hardware_lock_excludes_merge(tmp_path, sources):
    with HardwareLock(paths.hardware_lock(tmp_path)) as lock:
        lock.acquire('recording')
        with pytest.raises(HardwareBusy):
            merge.merge_datasets('fake', ['one', 'two'], 'new', tmp_path)


def test_fake_rejects_unmarked_metadata(tmp_path, sources):
    (sources[1] / 'meta/info.json').write_text('{"total_episodes": 3}')
    with pytest.raises(ValueError, match='not a fake'):
        merge.check_merge('fake', ['one', 'two'], 'new', tmp_path)


def test_real_requires_pinned_lerobot_and_v3(tmp_path, monkeypatch):
    monkeypatch.setattr(merge, 'version', lambda _: '0.6.0')
    with pytest.raises(ValueError, match='0.6.1'):
        merge.check_merge('real', ['one', 'two'], 'new', tmp_path)
    monkeypatch.setattr(merge, 'version', lambda _: '0.6.1')
    for name in ('one', 'two'):
        root = fake(tmp_path, name)
        target = tmp_path / 'raw' / name
        target.parent.mkdir(exist_ok=True)
        root.rename(target)
        (target / 'meta/info.json').write_text('{"total_episodes":2,"codebase_version":"v2.1"}')
    with pytest.raises(ValueError, match='v3.0'):
        merge.check_merge('real', ['one', 'two'], 'new', tmp_path)


def test_reserved_prepared_name_and_staging_symlink(tmp_path, sources):
    (tmp_path / 'datasets/reserved').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        merge.check_merge('fake', ['one', 'two'], 'reserved', tmp_path)
    (sources[0].parent / '.merging').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='staging'):
        merge.merge_datasets('fake', ['one', 'two'], 'new', tmp_path)


def test_api_job_merges_and_opens_output(tmp_path, sources, monkeypatch):
    monkeypatch.setenv('MTREA_HARDWARE', 'fake')
    with TestClient(create_app(data_base=tmp_path)) as client:
        body = {'backend': 'fake', 'sources': ['one', 'two'], 'name': 'combined'}
        assert client.post('/api/datasets/merge', json=body).status_code == 422
        check = client.post('/api/datasets/merge/check', json=body)
        assert check.status_code == 200, check.text
        start = client.post('/api/datasets/merge', json=check.json())
        assert start.status_code == 202, start.text
        job_id = start.json()['job']['id']
        for _ in range(100):
            job = client.get(f'/api/jobs/{job_id}').json()
            if job['status'] != 'running':
                break
            time.sleep(.05)
        log = client.get(f'/api/jobs/{job_id}/logs').text
        assert job['status'] == 'done', log
        assert job['spec']['kind'] == 'merge'
        result = client.get('/api/datasets/fake/combined')
        assert result.status_code == 200
        assert result.json()['dataset']['episodes'] == 5
        assert client.post('/api/datasets/merge/check', json=body).status_code == 422


def test_api_rejects_changed_review_and_active_collection(tmp_path, sources, monkeypatch):
    monkeypatch.setenv('MTREA_HARDWARE', 'fake')
    app = create_app(data_base=tmp_path)
    with TestClient(app) as client:
        body = {'backend': 'fake', 'sources': ['one', 'two'], 'name': 'new'}
        checked = client.post('/api/datasets/merge/check', json=body).json()
        (sources[0] / 'changed').write_text('new data')
        assert client.post('/api/datasets/merge', json=checked).status_code == 422
        monkeypatch.setattr(app.state.collection, 'is_active', lambda: True)
        assert client.post('/api/datasets/merge', json=checked).status_code == 409
