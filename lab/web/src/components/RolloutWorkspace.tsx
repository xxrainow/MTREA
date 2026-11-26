import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadRolloutLog } from '../api/rollouts';
import type { useRollouts } from '../hooks/useRollouts';
import type { useRolloutValidations } from '../hooks/useRolloutValidations';
import RolloutCamera from './RolloutCamera';
import ValidationAttemptReview from './ValidationAttemptReview';
import NextValidationAttempt from './NextValidationAttempt';
import { validationStateLabels } from '../api/rolloutValidations';
import type { RolloutResultTarget } from '../rollouts/results';
import './RolloutWorkspace.css';

type Props = {
  rollout: ReturnType<typeof useRollouts>;
  validations: ReturnType<typeof useRolloutValidations>;
  open: boolean;
  onClose: () => void;
  onKeepOpen: () => void;
  onOpenResults: (target: RolloutResultTarget) => void;
};

export default function RolloutWorkspace({ rollout, validations, open, onClose, onKeepOpen, onOpenResults }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [logs, setLogs] = useState({ id: '', content: '', error: '' });
  const [dirtyReview, setDirtyReview] = useState({ id: '', dirty: false });
  const selected = rollout.selected;
  const status = selected?.status;
  const id = selected?.job.id ?? '';
  const validation = validations.items?.find(item => item.attempts.some(attempt => attempt.rollout.job.id === id));
  const attempt = validation?.attempts.find(item => item.rollout.job.id === id);
  const unsaved = dirtyReview.id === id && dirtyReview.dirty;
  const canClose = !rollout.owner && !rollout.busy && !validations.busy && !unsaved;
  const canNext = validation?.can_next && !rollout.owner && !rollout.busy && !validations.busy
    && !rollout.pollError && !validations.pollError && !unsaved;
  const canStop = Boolean(selected?.owner_alive) && ['connecting', 'running', 'stopping'].includes(selected?.state ?? '')
    && !rollout.busy;
  const canRelease = Boolean(selected?.owner_alive) && ['stopped', 'failed', 'aborted'].includes(selected?.state ?? '')
    && !rollout.busy && !rollout.pollError;
  const previewActive = Boolean(selected?.owner_alive && selected.state === 'running' && status?.preview_active
    && !status.preview_error && !rollout.pollError);
  const cameraNames = Object.keys(status?.frames ?? {});

  useEffect(() => {
    const element = dialog.current;
    // Losing an owner must not dismiss its errors or last known arm state.
    if (open && element && !element.open) element.showModal();
  }, [open]);

  useEffect(() => {
    const element = dialog.current;
    return () => element?.close();
  }, []);

  useEffect(() => {
    const element = dialog.current;
    if (element?.open) {
      element.scrollTop = 0;
      element.querySelector<HTMLElement>('#rollout-workspace-title')?.focus({ preventScroll: true });
    }
  }, [id]);

  function close() {
    if (!canClose) return;
    dialog.current?.close();
    onClose();
  }

  function command(action: 'stop' | 'release') {
    onKeepOpen();
    void rollout.command(action);
  }

  const logActive = Boolean(selected?.owner_alive);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      try {
        const result = await loadRolloutLog(id, controller.signal);
        if (!controller.signal.aborted) setLogs({ id, content: result.content, error: '' });
      } catch (error) {
        if (!controller.signal.aborted) setLogs({ id, content: '', error: errorMessage(error) });
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, logActive ? 1000 : 3000);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [id, logActive]);

  return <dialog ref={dialog} className="rollout-workspace" aria-labelledby="rollout-workspace-title"
    aria-describedby="rollout-workspace-instruction" onCancel={event => { event.preventDefault(); close(); }}>
    <header className="rollout-workspace-header">
      <div>
        <h2 id="rollout-workspace-title" tabIndex={-1} autoFocus>Rollout execution</h2>
        <p id="rollout-workspace-instruction">{selected?.session?.task ?? String(selected?.job.spec.metadata?.instruction ?? '—')}</p>
        {validation && <p className="rollout-round">Attempt {attempt?.number} / {validation.session.target_count} · {validationStateLabels[validation.state]}</p>}
      </div>
      <span className="run-status" role="status">{selected?.state ?? 'Waiting for rollout'}</span>
      <button type="button" className="run-button" disabled={!canClose || !id} onClick={() => {
        if (!canClose || !id) return;
        close();
        onOpenResults({ validationId: validation?.id, jobId: id });
      }}>View results</button>
      <button type="button" className="run-button" disabled={!canClose} onClick={close}>Close workspace</button>
    </header>
    <div className="rollout-workspace-body">
      <dl className="rollout-workspace-policy">
        <div><dt>Checkpoint / policy</dt><dd>{selected?.session?.checkpoint ?? String(selected?.job.spec.metadata?.checkpoint ?? '—')}</dd></div>
        <div><dt>Time limit</dt><dd>{String(selected?.job.spec.metadata?.time_limit_s ?? '—')} s</dd></div>
        <div><dt>Maximum relative target</dt><dd>{String(selected?.job.spec.metadata?.max_relative_target ?? '—')}</dd></div>
      </dl>
      {rollout.pollError && <p role="alert">Cannot refresh rollout state: {rollout.pollError}. Last known state is shown.</p>}
      {rollout.actionError && <p role="alert">{rollout.actionError}</p>}
      {validations.pollError && <p role="alert">Cannot refresh validation: {validations.pollError}</p>}
      {validations.actionError && <p role="alert">{validations.actionError}</p>}
      {validation?.session.server_mode === 'hold' && <p className="run-note">Hold mode · communication testing with simulated responses; no model inference.</p>}
      {unsaved && <p className="run-note" role="status">Save or discard outcome edits before switching attempts, starting the next attempt or closing.</p>}
      {validation && <section className="rollout-attempts" aria-label="Validation attempts">
        <p>{validation.started_count} / {validation.session.target_count} started · {validation.released_count} released · {validation.judged_count} judged</p>
        <div>{validation.attempts.map(item => <button key={item.rollout.job.id} type="button" className="run-button"
          aria-pressed={item.rollout.job.id === id} disabled={Boolean(rollout.owner) || rollout.busy || validations.busy || unsaved}
          onClick={() => rollout.select(item.rollout.job.id)}>
          #{item.number} · {item.outcome} · {item.rollout.state}
        </button>)}</div>
      </section>}
      <div className="rollout-workspace-grid">
        <section className="run-panel" aria-labelledby="rollout-camera-heading">
          <div className="run-panel-header"><h2 id="rollout-camera-heading">Camera preview</h2>
            <span className="run-status">{previewActive ? 'Observation frames' : 'Inactive'}</span></div>
          {status?.preview_error && <p role="alert">Preview: {status.preview_error}</p>}
          {previewActive && id && cameraNames.length > 0 ? <div className="run-cameras">
            {cameraNames.map((name) => <RolloutCamera key={`${id}:${name}`} jobId={id} name={name} />)}
          </div> : <div className="run-camera-placeholder">{selected ? 'No live preview · policy is not streaming observations' : 'Start a rollout to view its observations'}</div>}
        </section>

        <section className="run-panel" aria-labelledby="rollout-status-heading">
          <div className="run-panel-header"><h2 id="rollout-status-heading">Run status</h2>
            <span className="run-status" role="status">{selected?.state ?? (rollout.items === null ? 'Loading…' : 'Not started')}</span></div>
          <dl className="run-details">
            <div><dt>Rollout ID</dt><dd>{id || '—'}</dd></div>
            <div><dt>Elapsed time</dt><dd>{status ? `${status.elapsed_s.toFixed(1)} s` : '—'}</dd></div>
            <div><dt>Received chunks / executed actions</dt><dd>{status?.chunks_received ?? 0} / {status?.actions_executed ?? 0}</dd></div>
            <div><dt>Queued actions</dt><dd>{status?.queue_size ?? '—'}</dd></div>
            <div><dt>Observation-to-chunk latency</dt><dd>{status?.last_latency_ms == null ? '—' : `${status.last_latency_ms.toFixed(1)} ms`}</dd></div>
            <div><dt>Queue stalls</dt><dd>{status?.stalls ?? 0}{status?.stalling ? ' · waiting for actions' : ''}</dd></div>
            <div><dt>Device ownership</dt><dd>{selected?.owner_alive ? 'Retained · Release required' : status?.arm_released ? 'Released' : 'Not confirmed'}</dd></div>
            <div><dt>End reason</dt><dd>{status?.reason ?? '—'}</dd></div>
          </dl>
          {selected?.state === 'ended_unknown' && <p role="alert">The owner process is unavailable. Stop and physical arm state could not be verified.</p>}
          {selected?.read_error && <p role="alert">{selected.read_error}</p>}
          {status?.error && <p role="alert">{status.error}</p>}
          {status?.release_error && <p role="alert">Release failed: {status.release_error}</p>}
          {rollout.notice && <p role="status" className="run-note">{rollout.notice}</p>}
        </section>
      </div>
      {validation && attempt && <ValidationAttemptReview key={id} validation={validation} attempt={attempt}
        validations={validations} blocked={rollout.busy || validations.busy || Boolean(validations.pollError)}
        onDirtyChange={dirty => setDirtyReview({ id, dirty })} />}
      <details className="run-panel rollout-workspace-logs">
        <summary>Execution logs</summary>
        {logs.id === id && logs.error && <p role="alert">{logs.error}</p>}
        <pre className="run-log run-execution-log">{!id ? 'No rollout selected.' : logs.id !== id ? 'Loading logs…' : logs.content || 'No log output yet.'}</pre>
      </details>
    </div>
    <footer className="rollout-workspace-footer">
      <div>
        <strong>{rollout.owner ? 'Robot ownership retained' : status?.arm_released ? 'Arm released' : 'Robot ownership not confirmed'}</strong>
        <p>{rollout.owner ? 'Stop keeps torque and ownership. Release arm is required before closing.'
          : status?.arm_released ? validation?.can_next
            ? 'Reset the scene and confirm readiness for the next attempt, or close to pause validation.'
            : 'This execution has ended. Close to return to rollout settings.'
            : 'Inspect the recorded status. Physical arm state has not been verified.'}</p>
      </div>
      <div className="run-actions">
        {validation && !['completed', 'ended'].includes(validation.state) && validation.started_count < validation.session.target_count
          && <NextValidationAttempt key={`${validation.id}:${validation.started_count}`} validation={validation}
            enabled={Boolean(canNext)} onStart={() => { onKeepOpen(); void validations.next(validation); }} />}
        <button type="button" className="run-button rollout-stop" disabled={!canStop} onClick={() => command('stop')}>
          {rollout.pending?.action === 'stop' ? 'Waiting for Stop…' : 'Stop execution'}</button>
        <button type="button" className="run-button" disabled={!canRelease} onClick={() => command('release')}>
          {rollout.pending?.action === 'release' ? 'Waiting for Release…' : 'Release arm (torque off)'}</button>
      </div>
    </footer>
  </dialog>;
}
