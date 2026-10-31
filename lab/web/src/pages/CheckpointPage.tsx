import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadCheckpoints, loadRunConfig, loadRuns } from '../api/training';
import type { CheckpointsResponse } from '../api/training';
import { checkpointId, parseCheckpointId, sortCheckpoints, stateLabel } from '../training/model';
import type { CheckpointEntry, RunSummary } from '../training/model';
import './TrainPage.css';
import './CheckpointPage.css';

type Props = {
  // "<job id>:<step>" of a checkpoint to show in detail, e.g. from Train's View final checkpoint.
  selectedId: string | null;
  onSelect: (id: string | null) => void;
};

function hostPath(response: CheckpointsResponse, entry: CheckpointEntry) {
  const sep = response.run_dir.includes('\\') ? '\\' : '/';
  return `${response.run_dir}${sep}${entry.path.replaceAll('/', sep)}`;
}

export default function CheckpointPage({ selectedId, onSelect }: Props) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointsResponse | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    loadRuns(controller.signal)
      .then(next => { setRuns(next); setError(''); })
      .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)); });
    return () => controller.abort();
  }, [refresh]);

  const selected = selectedId ? parseCheckpointId(selectedId) : null;
  // A direct link from Train takes precedence over the last browsed run.
  const run = runs?.find(r => r.job.id === (selected?.jobId ?? selectedRunId)) ?? runs?.[0] ?? null;
  const runId = run?.job.id ?? null;

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    Promise.all([loadCheckpoints(runId, controller.signal), loadRunConfig(runId, controller.signal)])
      .then(([nextCheckpoints, nextConfig]) => {
        setCheckpoints(nextCheckpoints);
        setConfig(nextConfig.config);
        setError('');
      })
      .catch(reason => {
        if (controller.signal.aborted) return;
        setCheckpoints(null);
        setError(errorMessage(reason));
      });
    return () => controller.abort();
  }, [runId, refresh]);

  const entries = checkpoints && checkpoints.run_dir === run?.job.spec.run_dir
    ? sortCheckpoints(checkpoints.checkpoints?.checkpoints ?? []) : [];
  const detail = selected && selected.jobId === runId ? entries.find(c => c.step === selected.step) ?? null : null;
  const fake = run?.status?.fake ?? run?.job.spec.cmd.includes('--fake') ?? false;

  function selectRun(id: string) {
    setSelectedRunId(id);
    onSelect(null);
  }
  function showList() {
    if (runId) setSelectedRunId(runId);
    onSelect(null);
  }

  return <main className="train-page">
    <div className="train-heading">
      <div><h1>Checkpoint</h1><p>Browse checkpoints from each training run, as listed in its checkpoints.json.</p></div>
      <button type="button" onClick={() => setRefresh(k => k + 1)}>Refresh</button>
    </div>
    {error && <p className="train-errors" role="alert">{error}</p>}
    <div className="checkpoint-layout">
      <section className="train-panel checkpoint-runs" aria-label="Training runs">
        <div className="checkpoint-runs-heading"><h2>Training runs</h2><span className="train-badge">{runs?.length ?? 0}</span></div>
        <div className="checkpoint-run-list">
          {(runs ?? []).map(item => <button
            key={item.job.id} type="button" className="checkpoint-run"
            aria-pressed={runId === item.job.id} onClick={() => selectRun(item.job.id)}
          ><span>{item.job.spec.label}</span><small>{stateLabel(item.state, item.status?.failure)}{item.status?.fake ? ' · Fake' : ''} · {item.checkpoint_count ?? 0} checkpoints</small></button>)}
        </div>
        {runs !== null && !runs.length && <p className="train-empty checkpoint-runs-empty">No training runs yet.</p>}
      </section>
      <section className="train-panel checkpoint-content" aria-label="Selected run checkpoints">
        {!run ? <><h2>Saved checkpoints</h2><p className="train-empty">{runs === null ? 'Loading…' : 'Start a training run from Train to create one.'}</p></> : <>
          <div className="train-panel-heading"><h2>{run.job.spec.label}</h2><span className="train-run-badges">{fake && <span className="train-badge train-badge-fake">Fake</span>}<span className="train-badge">{stateLabel(run.state, run.status?.failure)}</span></span></div>
          <dl className="checkpoint-summary">
            <div><dt>Method</dt><dd>{run.status?.method ?? '—'}</dd></div>
            <div><dt>Seed</dt><dd>{run.status?.seed ?? '—'}</dd></div>
            <div><dt>Target</dt><dd>{run.job.spec.executor === 'ssh' ? `GPU host · ${run.job.spec.host ?? ''}` : 'This machine'}</dd></div>
            <div><dt>Job</dt><dd>{run.job.id}</dd></div>
          </dl>
          {run.unreachable && <p className="train-note" role="status">GPU host unreachable: {run.unreachable}</p>}
          {detail ? <>
            <button type="button" className="checkpoint-back" onClick={showList}>← Checkpoint list</button>
            <h2>Step {detail.step.toLocaleString()} · {detail.last ? 'Last' : 'Kept'}</h2>
            <dl className="train-details">
              <div><dt>Path on {run.job.spec.executor === 'ssh' ? 'the GPU host' : 'this machine'}</dt><dd className="train-path">{checkpoints && hostPath(checkpoints, detail)}</dd></div>
              <div><dt>Contents</dt><dd>{detail.has_training_state ? 'Full · weights and training state' : 'Weights only'}</dd></div>
              <div><dt>Complete</dt><dd>{detail.complete ? 'Yes' : 'No · still being written or cut off'}</dd></div>
              <div><dt>Saved at</dt><dd>{new Date(detail.saved_at).toLocaleString()}</dd></div>
              <div><dt>Git commit (run host)</dt><dd>{String(config?.git_commit ?? '—')}{config?.git_dirty ? ' · dirty' : ''}</dd></div>
            </dl>
            <details key={selectedId}><summary>Run configuration and provenance (config.json)</summary><pre className="train-log">{config ? JSON.stringify(config, null, 2) : 'config.json not written yet.'}</pre></details>
          </> : <>
            <div className="train-panel-heading"><h2>Saved checkpoints</h2><span className="train-badge">{entries.length}</span></div>
            <div className="train-checkpoints-table-wrapper"><table className="train-checkpoints-table checkpoint-table">
              <thead><tr><th scope="col">Checkpoint</th><th scope="col">Contents</th><th scope="col">Saved</th><th scope="col"><span className="train-sr-only">Actions</span></th></tr></thead>
              <tbody>{entries.map(c => <tr key={c.step}>
                <td>Step {c.step.toLocaleString()} {c.last ? <span className="checkpoint-final">Last</span> : <span className="checkpoint-intermediate">· Kept</span>}{!c.complete && <span className="checkpoint-intermediate"> · incomplete</span>}</td>
                <td>{c.has_training_state ? 'Full' : 'Weights only'}</td>
                <td>{new Date(c.saved_at).toLocaleString()}</td>
                <td><button type="button" onClick={() => onSelect(checkpointId(run.job.id, c.step))} aria-label={`View ${run.job.spec.label} step ${c.step}`}>View details</button></td>
              </tr>)}
              {!entries.length && <tr><td colSpan={4} className="train-empty">{run.state === 'running' || run.state === 'starting' ? 'No checkpoints yet. Waiting for the first save.' : 'This run has no checkpoint on disk.'}</td></tr>}
              </tbody>
            </table></div>
            <p className="train-note">Latest step first. Older checkpoints are pruned unless kept with --keep-checkpoints. Checkpoints stay where they were written; nothing is downloaded.</p>
          </>}
        </>}
      </section>
    </div>
    <p className="train-note">Only complete checkpoints may be evaluated. Evaluation uses the last checkpoint of every method. Training loss is not task success rate.</p>
  </main>;
}
