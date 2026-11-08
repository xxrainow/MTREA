import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadDatasets } from '../api/datasets';
import type { DatasetSummary } from '../api/datasets';
import { loadTargets, loadJobLog, stopJob } from '../api/training';
import type { Targets } from '../api/training';
import { loadPolicyCheckpoints, loadPolicySessions, openPolicyTunnel, startPolicy } from '../api/serving';
import type { PolicyCheckpoint, PolicySession } from '../api/serving';
import './RunPage.css';

const active = (status?: string) => status === 'running' || status === 'queued';
const label = (state: string) => ({ starting: 'Starting', loading: 'Loading model', ready: 'Model ready',
  serving: 'Client connected', stopped: 'Stopped', failed: 'Failed', ended_unknown: 'Ended · status unknown' }[state] ?? state);

function RunPage() {
  const [target, setTarget] = useState<'ssh' | 'local'>('ssh');
  const [mode, setMode] = useState<'real' | 'hold'>('real');
  const [source, setSource] = useState('');
  const [dataset, setDataset] = useState('');
  const [checkpoints, setCheckpoints] = useState<PolicyCheckpoint[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [targets, setTargets] = useState<Targets | null>(null);
  const [sessions, setSessions] = useState<PolicySession[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [optionsError, setOptionsError] = useState('');
  const [logs, setLogs] = useState('');
  const [tunnelLogs, setTunnelLogs] = useState('');
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([loadPolicyCheckpoints(controller.signal), loadDatasets(controller.signal), loadTargets(controller.signal)])
      .then(([next, data, hosts]) => {
        setCheckpoints(next.checkpoints); setOptionsError(next.errors.join('\n'));
        setDatasets(data.filter(d => d.backend === 'real')); setTargets(hosts);
      })
      .catch(reason => { if (!controller.signal.aborted) setOptionsError(errorMessage(reason)); });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await loadPolicySessions(controller.signal);
        if (controller.signal.aborted) return;
        setSessions(next); setLoaded(true); setPollError('');
      } catch (reason) {
        if (!controller.signal.aborted) setPollError(errorMessage(reason));
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refresh]);

  const session = sessions.find(s => s.job.id === selectedId) ?? sessions[0] ?? null;
  const sessionId = session?.job.id;
  const tunnelId = session?.tunnel?.id;
  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const [serverLog, tunnelLog] = await Promise.all([
          loadJobLog(sessionId!, controller.signal),
          tunnelId ? loadJobLog(tunnelId, controller.signal) : Promise.resolve({ content: '' }),
        ]);
        if (!controller.signal.aborted) { setLogs(serverLog.content); setTunnelLogs(tunnelLog.content); }
      } catch (reason) {
        if (!controller.signal.aborted) setLogs(`Cannot read logs: ${errorMessage(reason)}`);
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [sessionId, tunnelId]);

  async function perform(action: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await action(); setRefresh(n => n + 1); }
    catch (reason) { setError(errorMessage(reason)); }
    finally { setBusy(false); }
  }
  const running = sessions.some(s => active(s.job.status));
  const canStart = loaded && !pollError && !running && !busy && (target === 'ssh'
    ? Boolean(source && targets?.ssh.reachable) : Boolean(dataset));
  const currentActive = active(session?.job.status);
  const tunnelActive = active(session?.tunnel?.status);

  return <main className="run-page">
    <div className="run-heading"><h1>Run &amp; Evaluate</h1><p>Prepare a policy server and inspect its connection before running the robot.</p></div>
    {error && <p className="run-error" role="alert">{error}</p>}
    {pollError && <p className="run-error" role="alert">Cannot update session status: {pollError}. Displayed values may be out of date.</p>}
    <div className="run-layout">
      <div className="run-main">
        <section className="run-panel" aria-label="Policy session">
          <div className="run-panel-header"><h2>Policy server</h2><span className="run-status">{session ? label(session.state) : loaded ? 'Not started' : 'Loading sessions…'}</span></div>
          <label className="run-field">Session<select value={sessionId ?? ''} onChange={event => setSelectedId(event.target.value)}>
            {!sessions.length && <option value="">No sessions yet</option>}
            {sessions.map(s => <option key={s.job.id} value={s.job.id}>{s.job.spec.label} · {label(s.state)}</option>)}
          </select></label>
          {session ? <>
            {session.unreachable && <p className="run-error" role="status">Cannot reach the GPU host: {session.unreachable}. The server may still be running.</p>}
            {session.status?.fake && <p className="run-note">Fake hold · repeats the observed joint positions. No model predictions.</p>}
            <dl className="run-details">
              <div><dt>Model</dt><dd>{label(session.state)}{session.unreachable ? ' · last known state' : ''}</dd></div>
              <div><dt>Checkpoint</dt><dd>{session.status?.checkpoint ?? String(session.job.spec.metadata?.checkpoint ?? 'Waiting for session files')}</dd></div>
              <div><dt>Host</dt><dd>{session.job.spec.host ?? 'This machine'}</dd></div>
              <div><dt>SSH tunnel</dt><dd>{session.job.spec.executor === 'local' ? 'Not needed for local fake hold' : tunnelActive ? 'SSH process running · link check required' : session.tunnel ? `Not running · ${session.tunnel.status}` : 'Not opened'}</dd></div>
              <div><dt>Robot PC endpoint</dt><dd>{session.client_address}</dd></div>
              <div><dt>Client</dt><dd>{!currentActive || session.unreachable ? 'Not confirmed' : session.status?.client_connected ? 'Connected' : 'Not connected'}</dd></div>
              <div><dt>Returned chunks</dt><dd>{session.status?.counts.chunks ?? 0}</dd></div>
              <div><dt>Prediction errors</dt><dd>{session.status?.counts.prediction_errors ?? 0}</dd></div>
            </dl>
            {session.status?.last_error && <p className="run-error" role="alert">Prediction error: {session.status.last_error}</p>}
            {session.status?.last_refusal && <p className="run-error" role="status">Last refused connection: {session.status.last_refusal}</p>}
            <div className="run-actions">
              {session.job.spec.executor === 'ssh' && <button className="run-button" disabled={busy || !currentActive || tunnelActive || Boolean(pollError) || Boolean(session.unreachable)} onClick={() => void perform(() => openPolicyTunnel(session.job.id))}>Open SSH tunnel</button>}
              <button className="run-button" disabled={busy || (!currentActive && !tunnelActive)} onClick={() => void perform(() => stopJob(session.job.id))}>Stop policy server</button>
            </div>
            <p className="run-note">A running tunnel is not a completed link test. Run serve_client_check from the robot PC before the first rollout. Model loading can take about 3.5 minutes.</p>
          </> : <p className="run-description">Choose a checkpoint or local fake hold to prepare a server. No robot hardware is opened.</p>}
        </section>
        <section className="run-panel"><h2>Server logs</h2><pre className="run-log">{session ? logs || 'Waiting for logs…' : 'No session selected.'}</pre>
          {tunnelId && <><h3>SSH tunnel logs</h3><pre className="run-log">{tunnelLogs || 'No tunnel output.'}</pre></>}
        </section>
        <section className="run-panel"><h2>Robot execution</h2><p className="run-description">Robot start/stop controls, live cameras and rollout results will be connected after the link and stop behavior are verified. Preparing a server does not move the arm.</p></section>
      </div>
      <section className="run-panel" aria-label="Prepare policy">
        <div className="run-panel-header"><h2>Prepare policy</h2><button className="run-button" disabled={busy} onClick={() => setRefresh(n => n + 1)}>Refresh</button></div>
        <label className="run-field">Run on<select value={target} disabled={busy} onChange={event => { const next = event.target.value as 'ssh' | 'local'; setTarget(next); if (next === 'local') setMode('hold'); }}>
          <option value="ssh">GPU host</option><option value="local">This machine · fake hold only</option>
        </select></label>
        <label className="run-field">Mode<select value={mode} disabled={busy || target === 'local'} onChange={event => setMode(event.target.value as 'real' | 'hold')}>
          <option value="real">Trained model</option><option value="hold">Fake hold · no model predictions</option>
        </select></label>
        {target === 'ssh' ? <>
          <label className="run-field">Checkpoint<select value={source} disabled={busy} onChange={event => setSource(event.target.value)}>
            <option value="">Select a checkpoint</option>{checkpoints.map(c => <option key={c.job_id} value={c.job_id}>{c.label} · step {c.step}</option>)}
          </select></label>
          <p className="run-description">Complete last checkpoints from real GPU training runs. Checkpoints stay on the host.</p>
          <p className="run-note">GPU host: {targets?.ssh.reachable ? targets.ssh.host : targets?.ssh.message ?? 'Checking…'}</p>
        </> : <>
          <label className="run-field">Recorded dataset<select value={dataset} disabled={busy} onChange={event => setDataset(event.target.value)}>
            <option value="">Select camera and joint settings</option>{datasets.map(d => <option key={d.id} value={d.name}>{d.name}</option>)}
          </select></label>
          <p className="run-description">Uses the dataset's camera and joint settings. Requires LeRobot 0.6.1 with async support in the backend Python environment.</p>
        </>}
        {optionsError && <p className="run-error" role="status">{optionsError}</p>}
        <div className="run-actions"><button className="run-button run-button-primary" disabled={!canStart} onClick={() => void perform(async () => {
          const job = await startPolicy({ target, mode, source_job_id: target === 'ssh' ? source : null, dataset_name: target === 'local' ? dataset : null });
          setSelectedId(job.id);
        })}>{busy ? 'Working…' : 'Start policy server'}</button></div>
        {running && <p className="run-note">A policy server is already active. Stop it before starting another. Training and policy serving cannot overlap.</p>}
      </section>
    </div>
  </main>;
}
export default RunPage;
