"""Immutable result bundles; original trials and annotations are never modified."""

import json
from datetime import datetime, timezone

from lab.core.recording_telemetry import atomic_write
from lab.core.models import JobKind
from lab.core.rollout_validation import identifier
from lab.core.runner import JobConflict


class RolloutResultMerge:
    def __init__(self, validations, archive):
        self.validations = validations
        self.archive = archive
        self.root = validations.root.parent / 'merged'

    def directory(self, value):
        path = self.root / identifier(value)
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError('Result directory is outside Lab storage')
        return path

    def snapshot(self, value, runner):
        try:
            manifest = json.loads((self.directory(value) / 'manifest.json').read_text())
        except FileNotFoundError:
            raise KeyError(value) from None
        attempts = [{**item, 'rollout': self.validations.rollout.snapshot(
            self.validations.rollout.get_job(item['job_id'], runner))} for item in manifest['attempts']]
        return {**manifest, 'attempts': attempts}

    def list(self, runner):
        if not self.root.exists():
            return []
        return [self.snapshot(path.name, runner) for path in self.root.iterdir()
                if path.is_dir() and (path / 'manifest.json').exists()]

    def source(self, ref, runner):
        if ref in self.archive.archived():
            raise JobConflict('Restore deleted results before merging')
        kind, value = ref['kind'], ref['id']
        if kind == 'merged':
            result = self.snapshot(value, runner)
            return result['condition'], result['attempts'], result['target_count']
        if kind == 'validation':
            result = self.validations.snapshot(value, runner)
            if result['state'] not in {'completed', 'ended'}:
                raise JobConflict('End validation and Release the arm before merging')
            attempts = result['attempts']
            target = result['session']['target_count']
        elif kind == 'standalone':
            job = self.validations.rollout.get_job(value, runner)
            context = job.spec.metadata.get('validation')
            if context and (self.validations.directory(context['id']) / 'session.json').exists():
                raise JobConflict('Select the whole validation result')
            attempts = [{'number': 1, 'rollout': self.validations.rollout.snapshot(job),
                         'outcome': 'unjudged', 'notes': ''}]
            target = 1
        else:
            raise ValueError('Invalid source kind')
        conditions = []
        for item in attempts:
            trial = item['rollout']
            meta, session = trial['job']['spec']['metadata'], trial['session'] or {}
            config = session.get('configuration') or {}
            server = runner.store.get(meta.get('server_job_id', ''))
            mode = server.spec.metadata.get('mode', 'unknown') if server else 'unknown'
            conditions.append({
                'checkpoint': session.get('checkpoint', meta.get('checkpoint')),
                'instruction': session.get('task', meta.get('instruction')),
                'time_limit_s': session.get('time_limit_s', meta.get('time_limit_s')),
                'max_relative_target': config.get('robot', {}).get('max_relative_target', meta.get('max_relative_target')),
                'mode': mode, 'server_target': server.spec.executor.value if server else None, 'unknown_server': meta.get('server_job_id') if mode == 'unknown' else None,
                'robot_id': session.get('robot_id'), 'lerobot_version': session.get('lerobot_version'),
                'client': {key: val for key, val in config.items()
                           if key not in {'task', 'server_address', 'pretrained_name_or_path'}},
            })
        if not conditions or any(condition != conditions[0] for condition in conditions):
            raise JobConflict('Sources must contain attempts with the same recorded execution settings')
        return conditions[0], attempts, target

    def create(self, refs, name, request_id, runner):
        path = self.directory(request_id) / 'manifest.json'
        if not isinstance(name, str) or not name.strip() or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ValueError('Enter a result name of 1–120 characters')
        if not 2 <= len(refs) <= 50 or len({(r['kind'], r['id']) for r in refs}) != len(refs):
            raise ValueError('Select 2–50 distinct results')
        intent = {'sources': refs, 'name': name.strip()}
        if path.exists():
            saved = self.snapshot(request_id, runner)
            if saved['intent'] != intent:
                raise JobConflict('Request ID already belongs to another merge')
            return saved
        runner.refresh(kind=JobKind.ROLLOUT)
        sources = [self.source(ref, runner) for ref in refs]
        condition = sources[0][0]
        if any(source[0] != condition for source in sources):
            raise JobConflict('Merge requires the same checkpoint, instruction and recorded execution settings')
        if any(condition[key] is None for key in ('checkpoint', 'instruction', 'time_limit_s', 'max_relative_target')):
            raise JobConflict('Execution settings are missing; these results cannot be merged')
        attempts, seen = [], set()
        for ref, (_, items, _) in zip(refs, sources):
            for item in items:
                trial = item['rollout']
                if trial['owner_alive'] or trial['read_error'] or trial['job']['status'] in {'running', 'queued'}:
                    raise JobConflict('Stop and Release every attempt before merging')
                job_id = trial['job']['id']
                if job_id in seen:
                    raise JobConflict('Selected results overlap: an attempt cannot be counted twice')
                seen.add(job_id)
                attempts.append({'number': len(attempts) + 1, 'job_id': job_id,
                                 'outcome': item['outcome'], 'notes': item['notes'],
                                 'source': {**ref, 'number': item['number']}})
        manifest = {'id': request_id, 'name': name.strip(), 'created_at': datetime.now(timezone.utc).isoformat(),
                    'condition': condition, 'target_count': sum(source[2] for source in sources),
                    'attempts': attempts, 'intent': intent, 'git_commit': runner._git_commit()}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(manifest, ensure_ascii=False).encode('utf-8'))
        return self.snapshot(request_id, runner)
