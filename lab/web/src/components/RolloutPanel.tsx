import { useState } from 'react';
import { errorMessage } from '../api/hardware';
import { checkValidation, validationStateLabels } from '../api/rolloutValidations';
import type { useRolloutValidations } from '../hooks/useRolloutValidations';
import type { PolicySession } from '../api/serving';
import { policySessionReady } from '../api/serving';
import type { useRollouts } from '../hooks/useRollouts';

export default function RolloutPanel({ sessions, sessionError, rollout, validations, serverId, onServerChange, onPrepareServer, onOpenWorkspace }: {
  sessions: PolicySession[] | null;
  sessionError: string;
  rollout: ReturnType<typeof useRollouts>;
  validations: ReturnType<typeof useRolloutValidations>;
  serverId: string;
  onServerChange: (id: string) => void;
  onPrepareServer: () => void;
  onOpenWorkspace: () => void;
}) {
  const [instruction, setInstruction] = useState('');
  const [targetLimit, setTargetLimit] = useState('');
  const [timeLimit, setTimeLimit] = useState('');
  const [attemptCount, setAttemptCount] = useState('');
  const active = validations.active;
  const config = active?.session.configuration;
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState('');
  const [checkMessage, setCheckMessage] = useState('');
  const [checkError, setCheckError] = useState('');
  const selected = rollout.selected;
  const id = selected?.job.id ?? '';
  const ownedServer = rollout.owner?.job.spec.metadata?.server_job_id;
  const ownerMetadata = rollout.owner?.job.spec.metadata;
  const readySessions = (sessions ?? []).filter(policySessionReady);
  const chosenId = (typeof ownedServer === 'string' ? ownedServer : '') || serverId;
  const server = readySessions.find((item) => item.job.id === chosenId);
  const body = { server_job_id: chosenId, instruction: instruction.trim(),
    max_relative_target: Number(targetLimit), time_limit_s: Number(timeLimit), target_count: Number(attemptCount) };
  const fingerprint = JSON.stringify(body);
  const blocked = rollout.busy || checking || Boolean(rollout.owner) || Boolean(active) || validations.busy;
  const valid = Boolean(server) && Boolean(body.instruction)
    && Number.isFinite(body.max_relative_target) && body.max_relative_target > 0
    && Number.isFinite(body.time_limit_s) && body.time_limit_s > 0
    && Number.isInteger(body.target_count) && body.target_count >= 1 && body.target_count <= 1000;
  const canCheck = valid && !blocked && rollout.items !== null && !rollout.pollError && !sessionError && validations.items !== null && !validations.pollError;
  const canStart = canCheck && checked === fingerprint;
  async function check() {
    if (!canCheck) return;
    setChecking(true);
    setCheckError('');
    setCheckMessage('');
    setChecked('');
    try {
      const result = await checkValidation(body);
      setChecked(fingerprint);
      setCheckMessage(`Settings checked · ${result.server_address} · ${result.checkpoint}`);
    } catch (error) { setCheckError(errorMessage(error)); }
    finally { setChecking(false); }
  }

  async function start() {
    if (!canStart) return;
    if (await validations.start(body)) {
      setChecked('');
      onOpenWorkspace();
    }
  }

  return <>
    <section className="run-panel" aria-labelledby="rollout-heading">
      <div className="run-panel-header">
        <h2 id="rollout-heading">Rollout settings</h2>
        <span className="run-status">On this machine</span>
      </div>
      <p className="run-description">Start connects to the real follower and cameras. Stop keeps torque and device ownership. Release arm turns torque off and disconnects.</p>
      <div className="run-execution-form">
        <label className="run-field">Policy server session
          <select value={chosenId} disabled={blocked} onChange={(event) => onServerChange(event.target.value)}>
            <option value="">Select a ready session</option>
            {chosenId && !readySessions.some((item) => item.job.id === chosenId)
              && <option value={chosenId}>{chosenId} · {rollout.owner ? 'current rollout server' : 'not ready'}</option>}
            {readySessions.map((item) => <option key={item.job.id} value={item.job.id}>
              {item.job.spec.label} · {item.job.spec.executor === 'local' ? 'This machine' : 'GPU host'}
              {item.job.spec.metadata?.mode === 'hold' ? ' · hold' : ' · real'}
            </option>)}
          </select>
        </label>
        <label className="run-field">Task instruction
          <textarea rows={2} value={config?.instruction ?? (rollout.owner ? String(ownerMetadata?.instruction ?? rollout.owner.session?.task ?? '') : instruction)}
            disabled={blocked} onChange={(event) => setInstruction(event.target.value)} />
        </label>
        <div className="run-number-fields">
          <label className="run-field">Maximum relative target
            <input type="number" min="0" step="any" value={config ? String(config.max_relative_target) : rollout.owner ? String(ownerMetadata?.max_relative_target ?? '') : targetLimit} disabled={blocked}
              onChange={(event) => setTargetLimit(event.target.value)} />
          </label>
          <label className="run-field">Time limit (seconds)
            <input type="number" min="0" step="any" value={config ? String(config.time_limit_s) : rollout.owner ? String(ownerMetadata?.time_limit_s ?? '') : timeLimit} disabled={blocked}
              onChange={(event) => setTimeLimit(event.target.value)} />
          </label>
        </div>
      </div>
      <label className="run-field">Number of attempts
        <input type="number" min="1" max="1000" step="1" value={active ? String(active.session.target_count) : attemptCount}
          disabled={blocked} onChange={event => setAttemptCount(event.target.value)} placeholder="e.g. 10" />
      </label>
      <p className="run-description">Each attempt starts manually after Release and scene reset. Trials are saved separately under one validation.</p>
      <p className="run-description">Joint limits use degrees; the gripper uses 0–100. Hold returns observed positions and still connects to the real arm.</p>
      {!server && <div className="run-note">
        <p>Prepare a ready policy server first. GPU sessions also need a running SSH tunnel.</p>
        <button type="button" className="run-button" onClick={onPrepareServer}>Manage policy servers</button>
      </div>}
      <div className="run-actions">
        <button className="run-button" type="button" disabled={!canCheck} onClick={() => void check()}>
          {checking ? 'Checking…' : 'Check rollout settings'}</button>
        <button className="run-button run-button-primary" type="button" disabled={!canStart} onClick={() => void start()}>
          {validations.busy ? 'Starting…' : 'Start validation'}</button>
      </div>
      {checkMessage && checked === fingerprint && <p role="status" className="run-note">{checkMessage}</p>}
      {checkError && <p role="alert">{checkError}</p>}
      {rollout.pollError && <p role="alert">Cannot refresh rollout state: {rollout.pollError}</p>}
      {rollout.actionError && <p role="alert">{rollout.actionError}</p>}
      {validations.pollError && <p role="alert">Cannot refresh validations: {validations.pollError}</p>}
      {validations.actionError && <p role="alert">{validations.actionError}</p>}
    </section>

    {validations.selected && <section className="run-panel" aria-label="Validation progress">
      <div className="run-panel-header"><h2>Validation</h2><span className="run-status">{validationStateLabels[validations.selected.state]}</span></div>
      <p>{validations.selected.started_count} / {validations.selected.session.target_count} attempts started · {validations.selected.released_count} released · {validations.selected.judged_count} judged</p>
      <p className="run-description">{validations.selected.session.configuration.instruction}</p>
      {validations.selected.start_errors.map(item => <p key={item.request_id} role="alert">Could not start attempt: {item.error}</p>)}
      {validations.selected.unresolved_requests.length > 0 && <p role="alert">An attempt start could not be verified. Inspect robot ownership before ending this validation.</p>}
      <div className="run-actions">
        <button type="button" className="run-button" disabled={validations.busy || !validations.selected.attempts.length}
          onClick={() => { validations.select(validations.selected!.id); onOpenWorkspace(); }}>Open validation workspace</button>
        {active && <button type="button" className="run-button" disabled={Boolean(rollout.owner) || rollout.busy || validations.busy || Boolean(validations.pollError)}
          onClick={() => void validations.end(active)}>End validation</button>}
      </div>
    </section>}

    <section className="run-panel" aria-labelledby="rollout-recent-heading">
      <div className="run-panel-header"><h2 id="rollout-recent-heading">Recent rollouts</h2>
        <span className="run-status">{selected?.state ?? (rollout.items === null ? 'Loading…' : 'Not started')}</span></div>
      {rollout.items?.length ? <>
        <label className="run-field">Rollout history
          <select value={id} disabled={Boolean(rollout.owner) || rollout.busy} onChange={event => rollout.select(event.target.value)}>
            {rollout.items?.map(item => <option key={item.job.id} value={item.job.id}>{item.job.spec.label} · {item.job.id}</option>)}
          </select>
        </label>
        <p className="run-description">{selected?.session?.task ?? String(selected?.job.spec.metadata?.instruction ?? '—')}</p>
        <div className="run-actions"><button type="button" className="run-button" onClick={onOpenWorkspace}>
          Open execution workspace
        </button></div>
      </> : <p className="run-description">Start a rollout to open its camera and execution workspace.</p>}
    </section>
  </>;
}
