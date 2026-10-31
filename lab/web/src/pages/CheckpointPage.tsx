import { useState } from 'react';
import type { Checkpoint, TrainingRun } from '../training/model';
import './TrainPage.css';
import './CheckpointPage.css';

type Props = {
  checkpoints: Checkpoint[];
  runs: TrainingRun[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
};
const statusLabels: Record<TrainingRun['status'], string> = {
  running: 'Running', disconnected: 'Connection lost', completed: 'Completed', stopped: 'Stopped',
};

export default function CheckpointPage({ checkpoints, runs, selectedId, onSelect }: Props) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const selected = checkpoints.find(c => c.id === selectedId);
  // A direct link from Train takes precedence over the last browsed run.
  const run = runs.find(item => item.id === (selected?.run.id ?? selectedRunId)) ?? runs.at(-1);
  const runCheckpoints = checkpoints.filter(c => c.run.id === run?.id).sort((a, b) => b.step - a.step);
  const counts = new Map<string, number>();
  checkpoints.forEach(c => counts.set(c.run.id, (counts.get(c.run.id) ?? 0) + 1));

  function selectRun(id: string) {
    setSelectedRunId(id);
    onSelect(null);
  }
  function showDetails(checkpoint: Checkpoint) {
    setSelectedRunId(checkpoint.run.id);
    onSelect(checkpoint.id);
  }
  function showList() {
    if (run) setSelectedRunId(run.id);
    onSelect(null);
  }

  return <main className="train-page">
    <div className="train-heading">
      <div><h1>Checkpoint</h1><p>Browse checkpoints from each training run.</p></div>
      <span className="train-badge">Simulation</span>
    </div>
    <div className="checkpoint-layout">
      <section className="train-panel checkpoint-runs" aria-label="Training runs">
        <div className="checkpoint-runs-heading"><h2>Training runs</h2><span className="train-badge">{runs.length}</span></div>
        <div className="checkpoint-run-list">
          {[...runs].reverse().map(item => <button
            key={item.id} type="button" className="checkpoint-run"
            aria-pressed={run?.id === item.id} onClick={() => selectRun(item.id)}
          ><span>{item.config.runName}</span><small>{statusLabels[item.status]} · {counts.get(item.id) ?? 0} checkpoints</small></button>)}
        </div>
        {!runs.length && <p className="train-empty checkpoint-runs-empty">No training runs yet.</p>}
      </section>
      <section className="train-panel checkpoint-content" aria-label="Selected run checkpoints">
        {!run ? <><h2>Saved checkpoints</h2><p className="train-empty">Start a training simulation from Train to create a run.</p></> : <>
          <div className="train-panel-heading"><h2>{run.config.runName}</h2><span className="train-badge">{statusLabels[run.status]}</span></div>
          <dl className="checkpoint-summary">
            <div><dt>Method</dt><dd>H0</dd></div><div><dt>Policy</dt><dd>π0.5</dd></div>
            <div><dt>Target</dt><dd>{run.config.target === 'ssh' ? 'GPU server' : 'Local'}</dd></div>
            <div><dt>Seed</dt><dd>{run.config.seed}</dd></div>
          </dl>
          {selected ? <>
            <button type="button" className="checkpoint-back" onClick={showList}>← Checkpoint list</button>
            <h2>Step {selected.step.toLocaleString()} · {selected.final ? 'Final' : 'Intermediate'}</h2>
            <dl className="train-details">
              {Object.entries({ Run: run.config.runName, Method: 'H0 · Full fine-tuning', Dataset: `${run.config.dataset} · T_sub · Simulated`, 'Base model': 'lerobot/pi05_base', Status: 'Simulated snapshot · No model file' }).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
            </dl>
            <details key={selected.id}><summary>Training configuration and provenance</summary><pre className="train-log">{JSON.stringify({ ...selected.run.config, runId: selected.run.id, gitCommit: null, simulation: true }, null, 2)}</pre><p className="train-note">A real run must record the execution host's commit and resolved configuration. This demo has no training artifacts.</p></details>
          </> : <>
            <div className="train-panel-heading"><h2>Saved checkpoints</h2><span className="train-badge">{runCheckpoints.length}</span></div>
            <div className="train-checkpoints-table-wrapper"><table className="train-checkpoints-table checkpoint-table">
              <thead><tr><th scope="col">Checkpoint</th><th scope="col">Status</th><th scope="col"><span className="train-sr-only">Actions</span></th></tr></thead>
              <tbody>{runCheckpoints.map(c => <tr key={c.id}>
                <td>Step {c.step.toLocaleString()} {c.final ? <span className="checkpoint-final">Final</span> : <span className="checkpoint-intermediate">· Intermediate</span>}</td>
                <td>Simulated</td><td><button type="button" onClick={() => showDetails(c)} aria-label={`View ${run.config.runName} step ${c.step}`}>View details</button></td>
              </tr>)}
              {!runCheckpoints.length && <tr><td colSpan={3} className="train-empty">{run.status === 'running' || run.status === 'disconnected' ? 'No checkpoints yet. Waiting for the first saved snapshot.' : 'This run ended without a saved checkpoint.'}</td></tr>}
              </tbody>
            </table></div>
            <p className="train-note">Simulation · No model files. Latest step first.</p>
          </>}
        </>}
      </section>
    </div>
    <p className="train-note">Demo snapshots disappear on reload. Training loss is not task success rate.</p>
  </main>;
}
