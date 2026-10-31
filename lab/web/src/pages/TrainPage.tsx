import { useState } from 'react';
import type { Dispatch } from 'react';
import { isTrainingActive } from '../training/model';
import type { TrainingAction, TrainingConfig, TrainingState } from '../training/model';
import './TrainPage.css';
import TrainingMetricChart from '../components/TrainingMetricChart';

type Props = { state: TrainingState; dispatch: Dispatch<TrainingAction>; onOpenCheckpoint: (id: string) => void };
function TrainPage({ state, dispatch, onOpenCheckpoint }: Props) {
  const [confirmStop, setConfirmStop] = useState(false);
  const { config, run } = state;
  function configure(patch: Partial<TrainingConfig>) {
    dispatch({ type: 'configure', config: { ...config, ...patch } });
  }
  function numericField(key: keyof Pick<TrainingConfig, 'seed' | 'steps' | 'batchSize' | 'learningRate' | 'warmupSteps' | 'saveFreq' | 'logFreq'>, label: string) {
    return <label className="train-field">{label}<input type="number" value={config[key]} min={key === 'seed' ? 0 : key === 'learningRate' ? undefined : 1} step={key === 'learningRate' ? 'any' : 1} placeholder={key === 'seed' ? 'e.g. 0' : undefined} onChange={e => configure({ [key]: e.target.value })} /></label>;
  }
  const active = isTrainingActive(run);
  const disconnected = run?.status === 'disconnected';
  return <main className="train-page">
    <div className="train-heading">
      <div><h1>Train</h1><p>{run ? 'Follow the active run and its training progress.' : 'Choose a compute target and configure policy training.'}</p></div>
      {!run && <button type="button" className="train-button-primary" disabled={!state.checked} onClick={() => dispatch({ type: 'start', id: crypto.randomUUID() })}>Start training</button>}
    </div>
    <p className="train-demo-note"><strong>Simulation only.</strong> No local process, SSH command, dataset transfer or model file is created. State is retained across menu navigation, but not browser reload.</p>
    {run ? <div className="train-monitor-main">
      <section className="train-panel">
        <div className="train-panel-heading"><h2>{run.config.runName}</h2><span className="train-badge" role="status">{disconnected ? 'Connection lost · Status unknown' : run.status === 'running' ? 'Training · Simulated' : run.status === 'completed' ? 'Completed · Simulated' : 'Stopped · Simulated'}</span></div>
        <dl className="train-details"><div><dt>Compute target</dt><dd>{run.config.target === 'ssh' ? 'GPU server · A100 / MIG 3g.40gb' : 'Local · Backend machine'}</dd></div><div><dt>Progress</dt><dd>{run.step.toLocaleString()} / {Number(run.config.steps).toLocaleString()} steps · {Math.floor(run.step / Number(run.config.steps) * 100)}%</dd></div></dl>
        <progress className="train-progress" aria-label="Training progress" value={run.step} max={Number(run.config.steps)} />
        <dl className="train-details"><div><dt>Last report</dt><dd>{disconnected ? 'No updates since connection loss' : 'Updated by simulation controls'}</dd></div><div><dt>train/loss</dt><dd>{run.metrics.at(-1)?.loss.toFixed(4) ?? '—'} · Simulated</dd></div><div><dt>Learning rate</dt><dd>{run.metrics.at(-1)?.lr.toExponential(3) ?? '—'} · Simulated</dd></div></dl>
        {active ? <>
          <div className="train-actions">
            {run.config.target === 'ssh' && <button type="button" onClick={() => dispatch({ type: disconnected ? 'reconnect' : 'disconnect' })}>{disconnected ? 'Reconnect · Demo' : 'Disconnect · Demo'}</button>}
            <button type="button" disabled={disconnected} onClick={() => setConfirmStop(true)}>Stop training…</button>
          </div>
          {confirmStop && <div className="train-stop-confirm" role="group" aria-label="Confirm stopping training"><p>Stop this simulation? Saved snapshots remain. The current step is not saved automatically.</p><div className="train-actions"><button type="button" disabled={disconnected} onClick={() => { dispatch({ type: 'stop' }); setConfirmStop(false); }}>Confirm stop</button><button type="button" onClick={() => setConfirmStop(false)}>Keep training</button></div></div>}
        </> : <div className="train-actions">{run.status === 'completed' && <button type="button" onClick={() => onOpenCheckpoint(`${run.id}:${Number(run.config.steps)}`)}>View final checkpoint</button>}<button type="button" className="train-button-primary" onClick={() => dispatch({ type: 'new' })}>New training setup</button></div>}
        <p className="train-note">{disconnected ? 'Connection loss is not training failure. Reconnect to the same run; do not submit another job.' : run.status === 'completed' ? 'Training is complete. Final metrics and logs remain here. Open the final checkpoint or choose New training setup when ready.' : 'Simulated metrics update every second while running, including on other pages. Checkpoints follow the configured save interval.'}</p>
      </section>
      <div className="train-grid"><TrainingMetricChart metrics={run.metrics} kind="loss" /><TrainingMetricChart metrics={run.metrics} kind="lr" /></div>
      <section className="train-panel"><h2>Training logs · Simulation</h2><pre className="train-log train-live-log">{run.logs.join('\n')}</pre></section>
    </div> : <>
      <section className="train-panel train-compute">
        <h2>Compute target</h2>
        <label className="train-field">Run training on<select value={config.target} onChange={e => configure({ target: e.target.value as TrainingConfig['target'] })}><option value="ssh">GPU server · Lab A100 · SSH</option><option value="local">Local · Backend machine</option></select></label>
        <p className="train-note">{config.target === 'ssh' ? 'Training will use a configured SSH server profile when remote execution is integrated.' : 'Local means the machine running the web backend, which may differ from your browser device.'}</p>
        <details><summary>{config.target === 'ssh' ? 'Server environment and connection settings' : 'Local execution requirements'}</summary>
          <dl className="train-details"><div><dt>Environment</dt><dd>Python 3.12+ / LeRobot 0.6.1</dd></div><div><dt>GPU</dt><dd>{config.target === 'ssh' ? 'A100 MIG 3g.40gb · Recipe target, not detected' : 'CUDA compatibility and memory must be checked'}</dd></div>{config.target === 'ssh' && <><div><dt>Connection profile</dt><dd>robot/configs/remote.yaml · Not loaded</dd></div><div><dt>Profile fields</dt><dd>host / user / ssh_key / repo_dir / data_dir / python</dd></div></>}</dl>
        </details>
      </section>
      <div className="train-grid">
        <section className="train-panel"><h2>Training inputs</h2>
          <label className="train-field">Dataset<select value={config.dataset} onChange={e => configure({ dataset: e.target.value })}><option value="">Select a dataset</option><option value="demo/pick_place">pick_place_demo · T_sub · Simulated</option></select></label>
          <p className="train-note">{config.target === 'ssh' ? 'Real local datasets will require transfer and verification on the GPU server.' : 'The execution machine must be able to read the training dataset.'}</p>
          <dl className="train-details"><div><dt>Training method</dt><dd>H0 · Full fine-tuning</dd></div><div><dt>Policy / Base model</dt><dd>π0.5 / lerobot/pi05_base</dd></div></dl>
          <label className="train-field">Run name<input value={config.runName} onChange={e => configure({ runName: e.target.value })} /></label>
          <p className="train-note">Planned output: data/runs/{config.runName || '<run_name>'}</p>
        </section>
        <section className="train-panel"><h2>Training conditions</h2>
          <div className="train-field-pair">{numericField('seed', 'Seed · Required')}{numericField('steps', 'Training steps')}{numericField('batchSize', 'Batch size')}{numericField('learningRate', 'Learning rate · Peak')}</div>
          <label className="train-checkbox"><input type="checkbox" checked={config.wandb} onChange={e => configure({ wandb: e.target.checked })} />Enable Weights & Biases</label>
          {config.wandb && <p className="train-note">Execution-host authentication must be checked before real training.</p>}
          <details><summary>Advanced settings</summary><div className="train-field-pair">{numericField('warmupSteps', 'Warmup steps')}{numericField('saveFreq', 'Checkpoint interval')}{numericField('logFreq', 'Log interval')}</div><dl className="train-details"><div><dt>Precision / Memory</dt><dd>bf16 / Gradient checkpointing</dd></div><div><dt>LR schedule</dt><dd>Cosine · Floor = peak / 10</dd></div><div><dt>Hub auto-upload</dt><dd>Off</dd></div></dl></details>
        </section>
      </div>
      <section className="train-panel train-checks"><h2>Pre-run checks</h2>
        <dl className="train-details"><div><dt>Input values</dt><dd role="status">{state.checked ? 'Valid for simulation' : 'Not validated'}</dd></div>{['Execution environment / Authentication', 'GPU allocation / Active job conflicts', 'Dataset access / T_sub role / Input compatibility', 'State and action quantiles · q01 / q99', 'Output directory / Disk / Execution-host commit'].map(label => <div key={label}><dt>{label}</dt><dd>Not checked · Backend integration pending</dd></div>)}</dl>
        {state.errors.length > 0 && <ul className="train-errors" role="alert">{state.errors.map(error => <li key={error}>{error}</li>)}</ul>}
        <div className="train-actions"><button type="button" onClick={() => dispatch({ type: 'check' })}>Validate demo settings</button></div><p className="train-note">Validation enables the top-right Start training button for simulation only. Editing any setting requires validation again.</p>
      </section>
    </>}
  </main>;
}
export default TrainPage;
