import { useEffect, useRef, useState } from 'react';
import { loadJobLog, loadMetrics, loadRunStatus, stopJob } from '../api/training';
import type { RunStatusResponse } from '../api/training';
import { errorMessage } from '../api/hardware';
import {
  appendMetrics, checkpointId, formatDuration, isActiveState, progressPercent, stateLabel,
} from '../training/model';
import type { Metric } from '../training/model';
import TrainingMetricChart from './TrainingMetricChart';
import '../pages/TrainPage.css';

const POLL_MS = 2000;

type Props = { jobId: string; onNew: () => void; onOpenCheckpoint: (id: string) => void };

// One run, read from its files on the executing machine; survives reload
// because nothing here is held only in the browser.
export default function TrainRunMonitor({ jobId, onNew, onOpenCheckpoint }: Props) {
  const [run, setRun] = useState<RunStatusResponse | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [log, setLog] = useState('');
  const [error, setError] = useState('');
  const [confirmStop, setConfirmStop] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [pollKey, setPollKey] = useState(0);
  const offset = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      try {
        const [next, page, tail] = await Promise.all([
          loadRunStatus(jobId, controller.signal),
          loadMetrics(jobId, offset.current, controller.signal),
          loadJobLog(jobId, controller.signal),
        ]);
        if (controller.signal.aborted) return;
        offset.current = page.offset;
        setRun(next);
        setMetrics(prev => appendMetrics(prev, page.metrics));
        setLog(tail.content);
        setError(next.job.unreachable ? `GPU host unreachable: ${next.job.unreachable}. Status is kept as last known.` : '');
        if (isActiveState(next.state)) timer = window.setTimeout(poll, POLL_MS);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(errorMessage(reason));
        timer = window.setTimeout(poll, POLL_MS);
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [jobId, pollKey]);

  async function stop() {
    setStopping(true);
    try {
      await stopJob(jobId);
      setConfirmStop(false);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setStopping(false);
      setPollKey(k => k + 1);
    }
  }

  const status = run?.status ?? null;
  const state = run?.state ?? 'starting';
  const active = isActiveState(state);
  const spec = run?.job.spec;
  const fake = status?.fake ?? spec?.cmd.includes('--fake') ?? false;
  const target = spec?.executor === 'ssh' ? `GPU host · ${spec.host ?? ''}` : 'This machine';
  const total = status?.total_steps;
  const lastCheckpoint = status?.last_checkpoint_step ?? null;

  return <div className="train-monitor-main">
    <section className="train-panel">
      <div className="train-panel-heading">
        <h2>{spec?.label ?? jobId}</h2>
        <span className="train-run-badges">
          {fake && <span className="train-badge train-badge-fake">Fake</span>}
          <span className="train-badge" role="status">{stateLabel(state, status?.failure)}</span>
        </span>
      </div>
      {error && <p className="train-errors" role="alert">{error}</p>}
      <dl className="train-details">
        <div><dt>Compute target</dt><dd>{target}</dd></div>
        <div><dt>Run directory</dt><dd className="train-path">{spec?.run_dir ?? '—'}</dd></div>
        <div><dt>Progress</dt><dd>{status ? `${status.step.toLocaleString()} / ${status.total_steps.toLocaleString()} steps · ${progressPercent(status.step, total)}%` : 'Waiting for status.json…'}</dd></div>
      </dl>
      <progress className="train-progress" aria-label="Training progress" value={status?.step ?? 0} max={total ?? 1} />
      <dl className="train-details">
        <div><dt>ETA</dt><dd>{active ? formatDuration(status?.eta_s) : '—'}{active && status?.eta_s != null ? ' · excludes checkpoint saves' : ''}</dd></div>
        <div><dt>train/loss</dt><dd>{status?.loss?.toFixed(4) ?? '—'}</dd></div>
        <div><dt>Learning rate</dt><dd>{status?.lr?.toExponential(3) ?? '—'}</dd></div>
        <div><dt>Seconds per step</dt><dd>{status?.seconds_per_step ?? '—'}</dd></div>
        <div><dt>Last checkpoint</dt><dd>{lastCheckpoint === null ? 'None yet' : `Step ${lastCheckpoint.toLocaleString()}`}</dd></div>
        {status?.failure && <div><dt>Failure</dt><dd>{status.failure === 'oom' ? 'Out of memory' : 'Error · see the log'}</dd></div>}
        {run?.job.exit_code != null && <div><dt>Exit code</dt><dd>{run.job.exit_code}</dd></div>}
      </dl>
      <div className="train-actions">
        {status?.wandb_url && <a className="train-link-button" href={status.wandb_url} target="_blank" rel="noreferrer">Open in W&B</a>}
        {active ? <button type="button" disabled={stopping} onClick={() => setConfirmStop(true)}>Stop training…</button>
          : <>
            {lastCheckpoint !== null && <button type="button" onClick={() => onOpenCheckpoint(checkpointId(jobId, lastCheckpoint))}>View final checkpoint</button>}
            <button type="button" className="train-button-primary" onClick={onNew}>New training setup</button>
          </>}
      </div>
      {confirmStop && active && <div className="train-stop-confirm" role="group" aria-label="Confirm stopping training">
        <p>Stop this run? Complete checkpoints remain; the current step is not saved. A stopped run cannot be resumed.</p>
        <div className="train-actions">
          <button type="button" disabled={stopping} onClick={() => void stop()}>{stopping ? 'Stopping…' : 'Confirm stop'}</button>
          <button type="button" onClick={() => setConfirmStop(false)}>Keep training</button>
        </div>
      </div>}
      <p className="train-note">{state === 'ended_unknown'
        ? 'The process ended without recording how. The files show the last known state.'
        : active ? 'Updated every 2 s from the run directory. The run continues on its machine while you are on other pages or reload.'
          : 'Final metrics and logs remain here. Open the final checkpoint or choose New training setup when ready.'}</p>
    </section>
    <div className="train-grid train-grid-3">
      <TrainingMetricChart metrics={metrics} kind="loss" fake={fake} />
      <TrainingMetricChart metrics={metrics} kind="lr" fake={fake} />
      <TrainingMetricChart metrics={metrics} kind="grad_norm" fake={fake} />
    </div>
    <section className="train-panel"><h2>Job log</h2><pre className="train-log train-live-log">{log || 'No output yet.'}</pre></section>
  </div>;
}
