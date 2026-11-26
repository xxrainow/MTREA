"""Manual repeated rollout validation, recovered from immutable sessions and events."""

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone

from lab.core.models import JobKind, RolloutConfig
from lab.core.rollout import RolloutService
from lab.core.runner import JobConflict


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{32}', value):
        raise ValueError('Invalid validation or request ID')
    return value


def validate_count(value):
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError('Number of attempts must be an integer from 1 to 1000')


def released(trial):
    status = trial['status'] or {}
    return (not trial['owner_alive'] and trial['job']['status'] not in {'queued', 'running'}
            and not trial['read_error'] and status.get('arm_released') is True
            and trial['state'] != 'ended_unknown')


class RolloutValidationService:
    def __init__(self, rollout: RolloutService):
        self.rollout = rollout
        self.root = rollout.root.parent / 'validation'

    def directory(self, validation_id):
        path = self.root / identifier(validation_id)
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError('Validation directory is outside Lab storage')
        return path

    def session(self, validation_id):
        try:
            return json.loads((self.directory(validation_id) / 'session.json').read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise KeyError(validation_id) from None

    def events(self, validation_id):
        path = self.directory(validation_id) / 'events.jsonl'
        if not path.exists():
            return []
        raw = path.read_bytes()
        # An interrupted final append is not a committed event. Refuse mutations
        # below so another append cannot silently join onto that partial line.
        lines = raw[:raw.rfind(b'\n') + 1].splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, validation_id, event):
        path = self.directory(validation_id) / 'events.jsonl'
        if path.exists() and path.stat().st_size:
            with path.open('rb') as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    raise ValueError('Validation event log has an incomplete append; inspect before continuing')
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({**event, 'time': now()}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self, validation_id, runner, jobs=None):
        session = self.session(validation_id)
        events = self.events(validation_id)
        jobs = runner.refresh(kind=JobKind.ROLLOUT) if jobs is None else jobs
        related = {job.spec.metadata['validation']['request_id']: job for job in jobs
                   if job.spec.kind == JobKind.ROLLOUT
                   and job.spec.metadata.get('validation', {}).get('id') == validation_id}
        attempts, unresolved, start_errors = [], [], []
        annotations = {event['job_id']: event for event in events if event['type'] == 'annotation'}
        failed = {event['request_id']: event for event in events if event['type'] == 'start_failed'}
        for event in events:
            if event['type'] != 'attempt_requested':
                continue
            token = event['request_id']
            job = related.get(token)
            if job is None:
                if token in failed:
                    start_errors.append(failed[token])
                unresolved.append(token)
                continue
            annotation = annotations.get(job.id, {})
            attempts.append({'number': event['number'], 'rollout': self.rollout.snapshot(job),
                             'outcome': annotation.get('outcome', 'unjudged'),
                             'notes': annotation.get('notes', '')})
        latest = attempts[-1]['rollout'] if attempts else None
        ended = any(event['type'] == 'ended' for event in events)
        safe = latest is None or released(latest)
        complete = len(attempts) >= session['target_count'] and safe and not unresolved
        state = ('ended' if ended else 'blocked' if unresolved else 'completed' if complete
                 else 'ready' if latest is None else 'awaiting_next' if safe
                 else 'running' if latest['state'] in {'running', 'connecting', 'stopping'}
                 else 'awaiting_release')
        return {'id': validation_id, 'session': session, 'state': state, 'attempts': attempts,
                'started_count': len(attempts), 'released_count': sum(released(a['rollout']) for a in attempts),
                'judged_count': sum(a['outcome'] != 'unjudged' for a in attempts),
                'can_next': not ended and not unresolved and safe and len(attempts) < session['target_count'],
                'unresolved_requests': unresolved, 'start_errors': start_errors}

    def list(self, runner):
        if not self.root.exists():
            return []
        jobs = runner.refresh(kind=JobKind.ROLLOUT)
        values = [self.snapshot(path.name, runner, jobs) for path in self.root.iterdir()
                  if path.is_dir() and (path / 'session.json').exists()]
        return sorted(values, key=lambda value: value['session']['created_at'], reverse=True)

    def create(self, config, target_count, request_id, runner, remote):
        validate_count(target_count)
        root = self.directory(request_id)
        if (root / 'session.json').exists():
            saved = self.session(request_id)
            if saved['configuration'] != asdict(config) or saved['target_count'] != target_count:
                raise JobConflict('This request ID already belongs to different settings')
            return self.snapshot(request_id, runner)
        if any(value['state'] not in {'completed', 'ended'} for value in self.list(runner)):
            raise JobConflict('Finish or end the current validation before starting another')
        checked = self.rollout.check(config, runner, remote)
        root.mkdir(parents=True, exist_ok=True)
        session = {'schema_version': 1, 'id': request_id, 'created_at': now(),
                   'configuration': asdict(config), 'target_count': target_count,
                   'checkpoint': checked['checkpoint'], 'server_address': checked['server_address'],
                   'instruction_key': hashlib.sha256(config.instruction.encode()).hexdigest(),
                   'server_mode': checked['session']['job']['spec']['metadata'].get('mode', 'unknown'),
                   'server_target': checked['session']['job']['spec']['executor'],
                   'git_commit': runner._git_commit()}
        with (root / 'session.json').open('x', encoding='utf-8') as stream:
            json.dump(session, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        return self.snapshot(request_id, runner)

    def next(self, validation_id, request_id, number, runner, remote, prepare_devices=None):
        identifier(request_id)
        if type(number) is not int or number < 1:
            raise ValueError('Attempt number must be a positive integer')
        value = self.snapshot(validation_id, runner)
        previous = next((event for event in self.events(validation_id)
                         if event.get('request_id') == request_id and event['type'] == 'attempt_requested'), None)
        if previous:
            if previous['number'] != number:
                raise JobConflict('This request ID already belongs to another attempt')
            return value
        if not value['can_next'] or number != value['started_count'] + 1:
            raise JobConflict('Wait for Release, then explicitly start the next available attempt')
        config = RolloutConfig(**value['session']['configuration'])
        checked = self.rollout.check(config, runner, remote)
        if checked['checkpoint'] != value['session']['checkpoint']:
            raise ValueError('The policy checkpoint changed; create a new validation')
        self.append(validation_id, {'type': 'attempt_requested', 'request_id': request_id, 'number': number})
        try:
            self.rollout.start(config, runner, remote, prepare_devices=prepare_devices,
                               expected_checkpoint=value['session']['checkpoint'],
                               validation={'id': validation_id, 'number': number, 'request_id': request_id})
        except Exception as exc:
            self.append(validation_id, {'type': 'start_failed', 'request_id': request_id, 'error': str(exc)})
            raise
        return self.snapshot(validation_id, runner)

    def annotate(self, validation_id, job_id, outcome, notes, runner):
        if outcome not in {'success', 'failure', 'unjudged'} or not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError('Invalid outcome or notes')
        value = self.snapshot(validation_id, runner)
        trial = next((a['rollout'] for a in value['attempts'] if a['rollout']['job']['id'] == job_id), None)
        if trial is None:
            raise KeyError(job_id)
        if trial['read_error'] or trial['state'] not in {'stopped', 'aborted', 'failed', 'ended_unknown'}:
            raise JobConflict('Stop the attempt before recording its outcome')
        self.append(validation_id, {'type': 'annotation', 'job_id': job_id, 'outcome': outcome, 'notes': notes})
        return self.snapshot(validation_id, runner)

    def end(self, validation_id, runner):
        value = self.snapshot(validation_id, runner)
        if any(a['rollout']['owner_alive'] or a['rollout']['job']['status'] in {'queued', 'running'}
               for a in value['attempts']):
            raise JobConflict('Stop and Release the current attempt before ending validation')
        if value['state'] not in {'completed', 'ended'}:
            self.append(validation_id, {'type': 'ended'})
        return self.snapshot(validation_id, runner)
