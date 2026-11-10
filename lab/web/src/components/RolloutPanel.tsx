import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { checkRollout, loadRolloutLog } from '../api/rollouts';
import type { PolicySession } from '../api/serving';
import { useRollouts } from '../hooks/useRollouts';
import RolloutCamera from './RolloutCamera';

export default function RolloutPanel({ sessions, sessionError, onServerInUse }: {
  sessions: PolicySession[] | null;
  sessionError: string;
  onServerInUse: (id: string | null) => void;
}) {
  const rollout = useRollouts();
  const [serverId, setServerId] = useState('');
  const [instruction, setInstruction] = useState('');
  const [targetLimit, setTargetLimit] = useState('');
  const [timeLimit, setTimeLimit] = useState('');
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState('');
  const [checkMessage, setCheckMessage] = useState('');
  const [checkError, setCheckError] = useState('');
  const [logs, setLogs] = useState({ id: '', content: '', error: '' });
  const selected = rollout.selected;
  const status = selected?.status;
  const id = selected?.job.id ?? '';
  const ownedServer = rollout.owner?.job.spec.metadata?.server_job_id;
  const ownerMetadata = rollout.owner?.job.spec.metadata;
  useEffect(() => {
    onServerInUse(typeof ownedServer === 'string' ? ownedServer : null);
    return () => onServerInUse(null);
  }, [ownedServer, onServerInUse]);

  const readySessions = (sessions ?? []).filter((item) =>
    ['ready', 'serving'].includes(item.state) && ['running', 'queued'].includes(item.job.status)
    && !item.unreachable && !item.read_error
    && (item.job.spec.executor === 'local' || item.tunnel?.status === 'running'));
  const chosenId = (typeof ownedServer === 'string' ? ownedServer : '') || serverId || readySessions[0]?.job.id || '';
  const server = readySessions.find((item) => item.job.id === chosenId);
  const body = { server_job_id: chosenId, instruction: instruction.trim(),
    max_relative_target: Number(targetLimit), time_limit_s: Number(timeLimit) };
  const fingerprint = JSON.stringify(body);
  const blocked = rollout.busy || checking || Boolean(rollout.owner);
  const valid = Boolean(server) && Boolean(body.instruction)
    && Number.isFinite(body.max_relative_target) && body.max_relative_target > 0
    && Number.isFinite(body.time_limit_s) && body.time_limit_s > 0;
  const canCheck = valid && !blocked && rollout.items !== null && !rollout.pollError && !sessionError;
  const canStart = canCheck && checked === fingerprint;
  const canStop = Boolean(selected?.owner_alive) && ['connecting', 'running', 'stopping'].includes(selected?.state ?? '')
    && !rollout.busy;
  const canRelease = Boolean(selected?.owner_alive) && ['stopped', 'failed', 'aborted'].includes(selected?.state ?? '')
    && !rollout.busy && !rollout.pollError;
  const previewActive = Boolean(selected?.owner_alive && selected.state === 'running' && status?.preview_active
    && !status.preview_error && !rollout.pollError);
  const cameraNames = Object.keys(status?.frames ?? {});

  async function check() {
    if (!canCheck) return;
    setChecking(true);
    setCheckError('');
    setCheckMessage('');
    setChecked('');
    try {
      const result = await checkRollout(body);
      setChecked(fingerprint);
      setCheckMessage(`Settings checked · ${result.server_address} · ${result.checkpoint}`);
    } catch (error) { setCheckError(errorMessage(error)); }
    finally { setChecking(false); }
  }

  async function start() {
    if (!canStart) return;
    if (await rollout.start(body)) setChecked('');
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

  return <>
    <section className="run-panel" aria-labelledby="rollout-heading">
      <div className="run-panel-header">
        <h2 id="rollout-heading">Robot execution</h2>
        <span className="run-status">On this machine</span>
      </div>
      <p className="run-description">Start connects to the real follower and cameras. Stop keeps torque and device ownership. Release arm turns torque off and disconnects.</p>
      <div className="run-execution-form">
        <label className="run-field">Policy server session
          <select value={chosenId} disabled={blocked} onChange={(event) => setServerId(event.target.value)}>
            <option value="">Select a ready session</option>
            {rollout.owner && !readySessions.some((item) => item.job.id === chosenId)
              && <option value={chosenId}>{chosenId} · current rollout server</option>}
            {readySessions.map((item) => <option key={item.job.id} value={item.job.id}>
              {item.job.spec.label} · {item.job.spec.executor === 'local' ? 'This machine' : 'GPU host'}
              {item.job.spec.metadata?.mode === 'hold' ? ' · hold' : ' · real'}
            </option>)}
          </select>
        </label>
        <label className="run-field">Task instruction
          <textarea rows={2} value={rollout.owner ? String(ownerMetadata?.instruction ?? rollout.owner.session?.task ?? '') : instruction}
            disabled={blocked} onChange={(event) => setInstruction(event.target.value)} />
        </label>
        <div className="run-number-fields">
          <label className="run-field">Maximum relative target
            <input type="number" min="0" step="any" value={rollout.owner ? String(ownerMetadata?.max_relative_target ?? '') : targetLimit} disabled={blocked}
              onChange={(event) => setTargetLimit(event.target.value)} />
          </label>
          <label className="run-field">Time limit (seconds)
            <input type="number" min="0" step="any" value={rollout.owner ? String(ownerMetadata?.time_limit_s ?? '') : timeLimit} disabled={blocked}
              onChange={(event) => setTimeLimit(event.target.value)} />
          </label>
        </div>
      </div>
      <p className="run-description">Joint limits use degrees; the gripper uses 0–100. Hold returns observed positions and still connects to the real arm.</p>
      {readySessions.length === 0 && <p className="run-note">Prepare a policy server first. GPU sessions also need a running SSH tunnel.</p>}
      <div className="run-actions">
        <button className="run-button" type="button" disabled={!canCheck} onClick={() => void check()}>
          {checking ? 'Checking…' : 'Check rollout settings'}</button>
        <button className="run-button run-button-primary" type="button" disabled={!canStart} onClick={() => void start()}>
          {rollout.posting && !selected?.owner_alive ? 'Starting…' : 'Start rollout'}</button>
      </div>
      {checkMessage && checked === fingerprint && <p role="status" className="run-note">{checkMessage}</p>}
      {checkError && <p role="alert">{checkError}</p>}
      {rollout.pollError && <p role="alert">Cannot refresh rollout state: {rollout.pollError}</p>}
      {rollout.actionError && <p role="alert">{rollout.actionError}</p>}
    </section>

    <section className="run-panel" aria-labelledby="rollout-status-heading">
      <div className="run-panel-header"><h2 id="rollout-status-heading">Run status</h2>
        <span className="run-status" role="status">{selected?.state ?? (rollout.items === null ? 'Loading…' : 'Not started')}</span></div>
      {Boolean(rollout.items?.length) && <label className="run-field">Rollout history
        <select value={id} disabled={Boolean(rollout.owner) || rollout.busy} onChange={(event) => rollout.select(event.target.value)}>
          {rollout.items?.map((item) => <option key={item.job.id} value={item.job.id}>{item.job.spec.label} · {item.job.id}</option>)}
        </select>
      </label>}
      <dl className="run-details">
        <div><dt>Rollout ID</dt><dd>{id || '—'}</dd></div>
        <div><dt>Instruction</dt><dd>{selected?.session?.task ?? String(selected?.job.spec.metadata?.instruction ?? '—')}</dd></div>
        <div><dt>Checkpoint / policy identifier</dt><dd>{selected?.session?.checkpoint ?? String(selected?.job.spec.metadata?.checkpoint ?? '—')}</dd></div>
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
      <div className="run-actions">
        <button type="button" className="run-button" disabled={!canStop} onClick={() => void rollout.command('stop')}>Stop execution</button>
        <button type="button" className="run-button" disabled={!canRelease} onClick={() => void rollout.command('release')}>Release arm (torque off)</button>
      </div>
      {rollout.notice && <p role="status" className="run-note">{rollout.notice}</p>}
      {selected?.owner_alive && <p className="run-note">Changing pages keeps this rollout connected. Return here to Stop or Release.</p>}
    </section>

    <section className="run-panel" aria-labelledby="rollout-camera-heading">
      <div className="run-panel-header"><h2 id="rollout-camera-heading">Camera preview</h2>
        <span className="run-status">{previewActive ? 'Observation frames' : 'Inactive'}</span></div>
      {status?.preview_error && <p role="alert">Preview: {status.preview_error}</p>}
      {previewActive && id && cameraNames.length > 0 ? <div className="run-cameras">
        {cameraNames.map((name) => <RolloutCamera key={`${id}:${name}`} jobId={id} name={name} />)}
      </div> : <div className="run-camera-placeholder">{selected ? 'No live preview · policy is not streaming observations' : 'Start a rollout to view its observations'}</div>}
    </section>

    <section className="run-panel" aria-labelledby="rollout-log-heading">
      <h2 id="rollout-log-heading">Execution logs</h2>
      {logs.id === id && logs.error && <p role="alert">{logs.error}</p>}
      <pre className="run-log run-execution-log">{!id ? 'No rollout selected.' : logs.id !== id ? 'Loading logs…' : logs.content || 'No log output yet.'}</pre>
    </section>
  </>;
}
