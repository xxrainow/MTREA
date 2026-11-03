"""Save renamed/relabelled dataset copies without changing source recordings."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from lab.core import dataset_manifest, paths
from lab.core.dataset_metadata import read_episode_tasks, read_jsonl, read_tasks
from lab.core.dataset_merge import new_dataset_path
from lab.core.datasets import Backend, dataset_root, get_dataset, validate_dataset_name
from lab.core.hardware.lock import HardwareLock


def check_edit(backend: Backend, source: str, name: str, instructions: dict[str, str],
               data_base: Path | None = None) -> dict:
    base = paths.root(data_base).resolve()
    validate_dataset_name(source)
    target = new_dataset_path(backend, name, base)
    root = dataset_root(backend, source, base)
    rows = dataset_manifest.listing(root)
    detail = get_dataset(backend, source, base)
    if detail.dataset.issues or not detail.episodes:
        raise ValueError('Dataset metadata must be healthy and contain saved episodes')
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if backend == 'fake' and info.get('fake') is not True:
        raise ValueError('Fake storage contains a non-fake dataset')
    if backend == 'real':
        if info.get('fake') or info.get('codebase_version') not in ('v2.0', 'v2.1', 'v3.0'):
            raise ValueError('Editing supports real LeRobot v2.0, v2.1 and v3.0 datasets')
        import pyarrow.parquet  # Fail preflight if the worker cannot edit Parquet metadata.
    if set(instructions) != set(detail.tasks):
        raise ValueError('Instructions changed since this form was opened; reload the dataset')
    mapping = {}
    for old, new in instructions.items():
        if not isinstance(new, str) or not new.strip() or len(new) > 2000:
            raise ValueError('Each instruction must contain 1–2000 characters')
        mapping[old] = new.strip()
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Instructions must remain distinct; duplicate task labels are not supported')
    if any(not set(ep.tasks).issubset(mapping) for ep in detail.episodes):
        raise ValueError('Episode instructions do not match the task table')
    return {'backend': backend, 'source': source, 'name': name, 'instructions': mapping,
            'source_version': dataset_manifest.fingerprint(rows), 'episodes': len(detail.episodes),
            'location': str(target.relative_to(base))}


def _rewrite_parquet(path: Path, mapping: dict[str, str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names) & {'task', 'tasks'}
    if not columns:
        parquet.close()
        return
    temporary = path.with_suffix('.editing')
    try:
        with pq.ParquetWriter(temporary, parquet.schema_arrow) as writer:
            for index in range(parquet.num_row_groups):
                table = parquet.read_row_group(index)
                for name in columns:
                    field = table.schema.field(name)
                    values = table[name].to_pylist()
                    values = ([mapping[value] for value in values] if name == 'task'
                              else [[mapping[value] for value in row] for row in values])
                    table = table.set_column(table.schema.get_field_index(name), field,
                                             pa.array(values, type=field.type))
                writer.write_table(table)
    finally:
        parquet.close()
    temporary.replace(path)


def rewrite_instructions(root: Path, mapping: dict[str, str]) -> None:
    meta = root / 'meta'
    v3 = meta / 'tasks.parquet'
    if v3.exists():
        import pandas as pd
        table = pd.read_parquet(v3)
        if table.index.name == 'task':
            table.index = table.index.map(lambda value: mapping[str(value)])
            table.index.name = 'task'
        if 'task' in table.columns:
            table['task'] = table['task'].map(lambda value: mapping[str(value)])
        table.to_parquet(v3)
        for path in (meta / 'episodes').rglob('*.parquet'):
            _rewrite_parquet(path, mapping)
    else:
        for filename, key in [('tasks.jsonl', 'task'), ('episodes.jsonl', 'tasks')]:
            path = meta / filename
            rows = read_jsonl(path)
            for row in rows:
                row[key] = mapping[row[key]] if key == 'task' else [mapping[value] for value in row[key]]
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    # Standard recordings store task_index only. Keep those files byte-for-byte.
    for path in (root / 'data').rglob('*.parquet'):
        _rewrite_parquet(path, mapping)


def edit_dataset(checked: dict, data_base: Path | None = None) -> dict:
    base = paths.root(data_base).resolve()
    lock = HardwareLock(paths.hardware_lock(base))
    lock.acquire('dataset-edit')
    stage = None
    try:
        current = check_edit(checked['backend'], checked['source'], checked['name'], checked['instructions'], base)
        if current['source_version'] != checked['source_version']:
            raise ValueError('Dataset changed after review; reload and check again')
        root = dataset_root(checked['backend'], checked['source'], base)
        print('PROGRESS phase=fingerprint', flush=True)
        before = dataset_manifest.fingerprint(dataset_manifest.manifest(root))
        target = new_dataset_path(checked['backend'], checked['name'], base)
        staging = target.parent / '.editing'
        if staging.is_symlink():
            raise ValueError('Edit staging directory cannot be a symbolic link')
        stage = staging / uuid4().hex
        stage.mkdir(parents=True)
        output = stage / 'output'
        print('PROGRESS phase=copy', flush=True)
        shutil.copytree(root, output)
        if dataset_manifest.fingerprint(dataset_manifest.manifest(output)) != before:
            raise ValueError('Copied files differ from the source; output was not published')
        print('PROGRESS phase=instructions', flush=True)
        if any(old != new for old, new in current['instructions'].items()):
            rewrite_instructions(output, current['instructions'])
        info_path = output / 'meta/info.json'
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if 'repo_id' in info:
            info['repo_id'] = f"local/{checked['name']}"
            info_path.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        print('PROGRESS phase=verify', flush=True)
        expected = {index: [current['instructions'][task] for task in tasks]
                    for index, tasks in read_episode_tasks(root).items()}
        if read_episode_tasks(output) != expected or read_tasks(output) != set(current['instructions'].values()):
            raise ValueError('Edited task metadata verification failed')
        dataset_manifest.listing(output)
        if dataset_manifest.fingerprint(dataset_manifest.manifest(root)) != before:
            raise ValueError('Source changed during edit; output was not published')
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                                text=True, capture_output=True, check=True).stdout.strip()
        record = {'source': checked['source'], 'name': checked['name'], 'backend': checked['backend'],
                  'source_fingerprint': before, 'instructions': current['instructions'], 'git_commit': commit,
                  'created_at': datetime.now(timezone.utc).isoformat()}
        with (output / 'meta/edits.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        new_dataset_path(checked['backend'], checked['name'], base)
        output.rename(target)
        print(f"EDITED backend={checked['backend']} name={checked['name']}", flush=True)
        return current
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        lock.release()


def edit_command(checked: dict, data_base: Path | None = None) -> str:
    base = paths.root(data_base).resolve()
    request_dir = base / 'jobs/edit-requests'
    if not request_dir.resolve().is_relative_to(base):
        raise ValueError('Request directory points outside data storage')
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f'{uuid4().hex}.json'
    with request_path.open('x', encoding='utf-8') as stream:
        json.dump(checked, stream, ensure_ascii=False)
    args = [sys.executable, '-m', 'lab.core.dataset_edit', f'--request={request_path}', f'--data-base={base}']
    return subprocess.list2cmdline(args) if os.name == 'nt' else shlex.join(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--data-base', type=Path, required=True)
    args = parser.parse_args()
    def stopped(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stopped)
    try:
        edit_dataset(json.loads(args.request.read_text(encoding='utf-8')), args.data_base)
    except Exception as exc:
        print(f'ERROR {exc}', flush=True)
        return 1
    print('STATUS done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
