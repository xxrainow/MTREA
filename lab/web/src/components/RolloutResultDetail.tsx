import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadRolloutLog, rolloutOwnsDevices } from '../api/rollouts';
import type { RolloutSnapshot } from '../api/rollouts';
import type { ValidationAttempt } from '../api/rolloutValidations';
import type { useRolloutResults } from '../hooks/useRolloutResults';
import type { ResultRun } from '../rollouts/results';
import { formatElapsed } from '../rollouts/results';
import ValidationAttemptReview from './ValidationAttemptReview';

function ResultLogs({ trial, refreshKey }: { trial: RolloutSnapshot; refreshKey: number }) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState('');
  const active = rolloutOwnsDevices(trial);
  const id = trial.job.id;
  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      try {
        const next = await loadRolloutLog(id, controller.signal);
        if (!controller.signal.aborted) { setContent(next.content); setError(''); }
      } catch (reason) {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      }
      if (active && !controller.signal.aborted) timer = window.setTimeout(poll, 2000);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [id, active, refreshKey]);
  return <details className="run-panel result-logs">
    <summary>Execution logs</summary>
    {error && <p role="alert">Cannot load execution logs: {error}</p>}
    <pre className="run-log run-execution-log">{content === null ? 'Loading logs…' : content || 'No saved log output.'}</pre>
  </details>;
}

export default function RolloutResultDetail({ run, attempt, results, navigationBlocked, onDirtyChange, onOpenExecution }: {
  run: ResultRun; attempt: ValidationAttempt; results: ReturnType<typeof useRolloutResults>;
  onDirtyChange: (dirty: boolean) => void; onOpenExecution: () => void;
  navigationBlocked: boolean;
}) {
  const trial = attempt.rollout;
  const status = trial.status;
  const active = rolloutOwnsDevices(trial);
  return <div className="result-attempt-detail">
    <section className="run-panel" aria-label="Attempt details">
      <div className="run-panel-header"><h2>Attempt {attempt.number}</h2><span className="run-status">{trial.state}</span></div>
      {active && <p className="run-note" role="status">This execution may still own the robot. Use Run & Evaluate for Stop and Release.</p>}
      {trial.state === 'ended_unknown' && <p role="alert">The owner process ended without a confirmed terminal state. Physical arm state is not verified.</p>}
      {trial.read_error && <p role="alert">Cannot read trial records: {trial.read_error}</p>}
      {status?.error && <p role="alert">Execution error: {status.error}</p>}
      {status?.release_error && <p role="alert">Release error: {status.release_error}</p>}
      {status?.preview_error && <p role="alert">Preview error: {status.preview_error}</p>}
      <dl className="run-details result-metrics">
        <div><dt>Rollout ID</dt><dd>{trial.job.id}</dd></div>
        <div><dt>Execution state</dt><dd>{trial.state}</dd></div>
        <div><dt>End reason</dt><dd>{status?.reason ?? 'Not recorded'}</dd></div>
        <div><dt>Device ownership</dt><dd>{active ? 'Retained or pending' : status?.arm_released ? 'Released' : 'Not confirmed'}</dd></div>
        <div><dt>Elapsed time</dt><dd>{formatElapsed(status?.elapsed_s)}</dd></div>
        <div><dt>Received chunks / executed actions</dt><dd>{status ? `${status.chunks_received ?? '—'} / ${status.actions_executed ?? '—'}` : '—'}</dd></div>
        <div><dt>Queued actions</dt><dd>{status?.queue_size ?? '—'}</dd></div>
        <div><dt>Last observation-to-chunk latency</dt><dd>{status?.last_latency_ms == null ? '—' : `${status.last_latency_ms.toFixed(1)} ms`}</dd></div>
        <div><dt>Queue stalls</dt><dd>{status?.stalls ?? '—'}{status?.stalling ? ' · waiting for actions' : ''}</dd></div>
        <div><dt>Exit code</dt><dd>{trial.job.exit_code ?? 'Not recorded'}</dd></div>
        <div><dt>Run commit</dt><dd>{trial.job.git_commit ?? 'Not recorded'}</dd></div>
      </dl>
      <p className="run-note">No playback video is recorded by the current rollout recorder. Live preview frames are temporary.</p>
      <div className="run-actions"><button type="button" className="run-button" disabled={navigationBlocked} onClick={onOpenExecution}>Open Run & Evaluate</button></div>
    </section>
    {run.validation ? <ValidationAttemptReview validation={run.validation} attempt={attempt}
      validations={{ busy: results.saving, annotate: results.annotate }} blocked={results.saving || Boolean(results.error)}
      onDirtyChange={onDirtyChange} />
      : <p className="run-note">{run.kind === 'unlinked' ? 'The validation record is unavailable. This saved attempt is read-only.' : 'This historical single rollout is read-only. No validation outcome record is available.'}</p>}
    {results.saveError?.jobId === trial.job.id && <p role="alert">Cannot save outcome: {results.saveError.message}</p>}
    <ResultLogs key={trial.job.id} trial={trial} refreshKey={results.refreshKey} />
    <details className="run-panel"><summary>Trial configuration and provenance</summary>
      <pre className="run-log">{JSON.stringify({ job: trial.job, session: trial.session }, null, 2)}</pre>
    </details>
  </div>;
}
