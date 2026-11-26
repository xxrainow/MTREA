"""Reversible result-list deletion; trial files and validation history stay intact."""

import json
import os
from datetime import datetime, timezone

from lab.core.models import JobKind
from lab.core.runner import JobConflict


class RolloutResultArchive:
    def __init__(self, validations):
        self.validations = validations
        self.path = validations.root.parent / 'result_archive.jsonl'

    def archived(self):
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        latest = {}
        for line in data[:data.rfind(b'\n') + 1].splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            latest[(event['kind'], event['id'])] = event
        return [{'kind': kind, 'id': identifier} for (kind, identifier), event in latest.items()
                if event['action'] == 'archive']

    def change(self, kind, identifier, action, runner):
        if kind not in {'validation', 'standalone', 'merged'} or action not in {'archive', 'restore'}:
            raise ValueError('Invalid result archive request')
        runner.refresh(kind=JobKind.ROLLOUT)
        if kind == 'merged':
            from lab.core.rollout_result_merge import RolloutResultMerge
            value = RolloutResultMerge(self.validations, self).snapshot(identifier, runner)
            trials = [item['rollout'] for item in value['attempts']]
        elif kind == 'validation':
            value = self.validations.snapshot(identifier, runner)
            if action == 'archive' and value['state'] not in {'completed', 'ended'}:
                raise JobConflict('End validation and Release the arm before deleting its result')
            trials = [item['rollout'] for item in value['attempts']]
        else:
            job = self.validations.rollout.get_job(identifier, runner)
            context = job.spec.metadata.get('validation')
            if context and (self.validations.directory(context['id']) / 'session.json').exists():
                raise JobConflict('Delete the whole validation result instead of an individual attempt')
            trials = [self.validations.rollout.snapshot(job)]
        if action == 'archive' and any(trial['owner_alive'] or trial['read_error']
                                      or trial['job']['status'] in {'running', 'queued'} for trial in trials):
            raise JobConflict('Stop and Release the rollout before deleting its result')
        already = any(item['kind'] == kind and item['id'] == identifier for item in self.archived())
        if already != (action == 'archive'):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open('rb') as stream:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b'\n':
                        raise ValueError('Result archive has an incomplete append; inspect before continuing')
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'kind': kind, 'id': identifier, 'action': action,
                                        'job_ids': [trial['job']['id'] for trial in trials],
                                        'time': datetime.now(timezone.utc).isoformat()}) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return {'archived': self.archived()}
