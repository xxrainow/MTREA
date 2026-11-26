import { useEffect, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { loadDatasets } from '../api/datasets';
import type { DatasetSummary } from '../api/datasets';
import { errorMessage } from '../api/hardware';
import {
  loadPolicyCheckpoints,
  loadPolicySessions,
  openPolicyTunnel,
  policySessionReady,
  startPolicy,
} from '../api/serving';
import type { PolicyCheckpoint, PolicySession } from '../api/serving';
import { loadJobLog, stopJob } from '../api/training';
import { validationStateLabels } from '../api/rolloutValidations';
import type { RolloutResultTarget } from '../rollouts/results';

import RolloutPanel from '../components/RolloutPanel';
import RolloutWorkspace from '../components/RolloutWorkspace';
import { useRollouts } from '../hooks/useRollouts';
import { useRolloutValidations } from '../hooks/useRolloutValidations';

import './RunPage.css';

type RunTab = 'policy' | 'rollout';
const tabKey = 'lab.run.tab';
const serverKey = 'lab.run.server';

function RunPage({ onOpenResults }: { onOpenResults: (target: RolloutResultTarget) => void }) {
  const [tab, setTab] = useState<RunTab>(() => sessionStorage.getItem(tabKey) === 'rollout' ? 'rollout' : 'policy');
  const [selectedServerId, setSelectedServerId] = useState(() => sessionStorage.getItem(serverKey) ?? '');
  const rollout = useRollouts();
  const validations = useRolloutValidations(rollout);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const ownedServer = rollout.owner?.job.spec.metadata?.server_job_id;
  const rolloutServerId = typeof ownedServer === 'string' ? ownedServer
    : validations.active?.session.configuration.server_job_id ?? null;

  function selectTab(next: RunTab) {
    sessionStorage.setItem(tabKey, next);
    setTab(next);
  }

  function selectServer(id: string) {
    if (rollout.owner || rollout.busy || validations.active || validations.busy) return;
    sessionStorage.setItem(serverKey, id);
    setSelectedServerId(id);
  }

  function handleTabKey(event: KeyboardEvent<HTMLButtonElement>) {
    let next: RunTab;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') next = tab === 'policy' ? 'rollout' : 'policy';
    else if (event.key === 'Home') next = 'policy';
    else if (event.key === 'End') next = 'rollout';
    else return;
    event.preventDefault();
    selectTab(next);
    document.getElementById(`run-tab-${next}`)?.focus();
  }
  const [target, setTarget] = useState<'ssh' | 'local'>('ssh');
  const [mode, setMode] = useState<'real' | 'hold'>('real');
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [checkpoint, setCheckpoint] = useState('');
  const [sessions, setSessions] = useState<PolicySession[] | null>(null);
  const [sessionError, setSessionError] = useState('');

  const [checkpoints, setCheckpoints] = useState<PolicyCheckpoint[] | null>(
    null,
  );
  const [checkpointError, setCheckpointError] = useState('');

  const [startingPolicy, setStartingPolicy] = useState(false);
  const [policyError, setPolicyError] = useState('');
  const [stoppingPolicyId, setStoppingPolicyId] = useState<string | null>(null);

  const [logJobId, setLogJobId] = useState<string | null>(null);
  const [logContent, setLogContent] = useState('');
  const [logError, setLogError] = useState('');
  const [logLoading, setLogLoading] = useState(false);

  const [openingTunnelId, setOpeningTunnelId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const next = await loadPolicySessions(controller.signal);

        if (controller.signal.aborted) return;

        setSessions(next);
        setSessionError('');
      } catch (reason) {
        if (!controller.signal.aborted) {
          setSessionError(errorMessage(reason));
        }
      }

      if (!controller.signal.aborted) {
        timer = setTimeout(poll, 2000);
      }
    }

    void poll();

    return () => {
      controller.abort();

      if (timer !== undefined) {
        clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    loadPolicyCheckpoints(controller.signal)
      .then((next) => {
        setCheckpoints(next.checkpoints);
        setCheckpointError(next.errors.join('\n'));
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setCheckpointError(errorMessage(reason));
        }
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    loadDatasets(controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) {
          setDatasets(next.filter((dataset) => dataset.backend === 'real'));
          setDatasetError('');
        }
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setDatasetError(errorMessage(reason));
        }
      });

    return () => controller.abort();
  }, []);

  const selectedServer = sessions?.find(item => item.job.id === (rolloutServerId ?? selectedServerId))
    ?? (rolloutServerId ? null : sessions?.find(policySessionReady)
      ?? sessions?.find(item => ['running', 'queued'].includes(item.job.status))) ?? null;
  const effectiveServerId = rolloutServerId ?? selectedServer?.job.id ?? '';
  const selectedCheckpoint = selectedServer?.status?.checkpoint
    ?? selectedServer?.job.spec.metadata?.checkpoint
    ?? rollout.owner?.session?.checkpoint
    ?? rollout.owner?.job.spec.metadata?.checkpoint;

  const policyRunning =
    sessions?.some(
      (session) =>
        session.job.status === 'running' || session.job.status === 'queued',
    ) ?? false;

  const canStartPolicy =
    (target === 'ssh'
      ? Boolean(checkpoints?.some((item) => item.job_id === checkpoint))
      : !datasetError && Boolean(datasets?.some((item) => item.name === datasetName))) &&
    sessions !== null &&
    !sessionError &&
    !policyRunning &&
    !startingPolicy && !validations.active && !validations.busy;

  async function handleStartPolicy() {
    if (!canStartPolicy) return;

    setStartingPolicy(true);
    setPolicyError('');

    try {
      const job = await startPolicy({
        target,
        mode,
        source_job_id: target === 'ssh' ? checkpoint : null,
        dataset_name: target === 'local' ? datasetName : null,
      });
      selectServer(job.id);
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setStartingPolicy(false);
    }
  }

  async function handleStopPolicy(jobId: string) {
    if (stoppingPolicyId !== null || rolloutServerId === jobId
      || rollout.busy || validations.busy || rollout.pollError || rollout.items === null) return;

    setStoppingPolicyId(jobId);
    setPolicyError('');

    try {
      await stopJob(jobId);
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setStoppingPolicyId(null);
    }
  }

  async function handleViewLogs(jobId: string) {
    if (logLoading) return;

    if (logJobId === jobId) {
      setLogJobId(null);
      return;
    }

    setLogJobId(jobId);
    setLogLoading(true);
    setLogContent('');
    setLogError('');

    try {
      const result = await loadJobLog(jobId);
      setLogContent(result.content);
    } catch (reason) {
      setLogError(errorMessage(reason));
    } finally {
      setLogLoading(false);
    }
  }

  async function handleOpenTunnel(jobId: string) {
    if (openingTunnelId !== null) return;

    setOpeningTunnelId(jobId);
    setPolicyError('');

    try {
      await openPolicyTunnel(jobId);
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setOpeningTunnelId(null);
    }
  }

  return (
    <main className="run-page">
      <div className="run-heading">
        <h1>Run & Evaluate</h1>
        <p>Run a trained policy and record rollout results.</p>
      </div>

      <div className="run-tabs" role="tablist" aria-label="Run and evaluate sections">
        {(['policy', 'rollout'] as const).map(item => <button key={item}
          type="button" role="tab" id={`run-tab-${item}`} aria-controls={`run-panel-${item}`}
          aria-selected={tab === item} tabIndex={tab === item ? 0 : -1}
          onKeyDown={handleTabKey} onClick={() => selectTab(item)}>
          {item === 'policy' ? 'Policy server' : 'Rollout execution'}
        </button>)}
      </div>

      <section className="run-panel run-context" aria-label="Selected policy server">
        <div className="run-panel-header">
          <h2>Selected policy</h2>
          <span className="run-status">{sessionError ? 'Server status unavailable' : selectedServer?.state ?? (sessions === null ? 'Loading…' : 'No session selected')}</span>
        </div>
        <dl className="run-context-details">
          <div><dt>Server session</dt><dd>{selectedServer?.job.spec.label ?? (effectiveServerId || 'Select or start a policy server')}</dd></div>
          <div><dt>Checkpoint / policy</dt><dd>{String(selectedCheckpoint ?? '—')}</dd></div>
          <div><dt>Target / mode</dt><dd>{selectedServer ? `${selectedServer.job.spec.executor === 'local' ? 'This machine' : 'GPU host'} · ${selectedServer.job.spec.metadata?.mode ?? '—'}` : '—'}</dd></div>
          <div><dt>SSH tunnel</dt><dd>{selectedServer ? selectedServer.job.spec.executor === 'local' ? 'Not required' : selectedServer.tunnel?.status ?? 'Not opened' : '—'}</dd></div>
        </dl>
        {sessionError && <p role="alert">{sessionError}</p>}
        {(selectedServer?.unreachable || selectedServer?.read_error) && <p role="alert">{selectedServer.unreachable || selectedServer.read_error}</p>}
        {rollout.pollError && <p role="alert">Cannot verify robot ownership: {rollout.pollError}</p>}
        {validations.pollError && <p role="alert">Cannot refresh validation: {validations.pollError}</p>}
        {validations.active && <div className="run-owner-notice">
          <span>Validation · {validations.active.started_count} / {validations.active.session.target_count} attempts · {validationStateLabels[validations.active.state]}. Policy selection stays fixed until validation ends.</span>
          <button type="button" className="run-button" disabled={validations.busy || rollout.busy} onClick={() => {
            selectTab('rollout');
            if (validations.active?.attempts.length) {
              validations.select(validations.active.id); setWorkspaceOpen(true);
            }
          }}>Continue validation</button>
        </div>}
        {rollout.owner && <div className="run-owner-notice" role="status">
          <span>Robot ownership retained · {rollout.owner.state}. Stop and Release are in Rollout execution.</span>
          {tab === 'policy' && <button type="button" className="run-button" onClick={() => selectTab('rollout')}>Open current rollout</button>}
        </div>}
      </section>

      <div className="run-layout" role="tabpanel" id="run-panel-policy" aria-labelledby="run-tab-policy" hidden={tab !== 'policy'}>
        <div className="run-main">
          <section className="run-panel">
            <h2>Policy server sessions</h2>

            {sessionError ? (
              <p role="alert">{sessionError}</p>
            ) : sessions === null ? (
              <p>Loading sessions…</p>
            ) : sessions.length === 0 ? (
              <p>No policy server sessions yet.</p>
            ) : (
              <ul className="run-sessions">
                {sessions.map((session) => (
                  <li key={session.job.id} className={`run-session${effectiveServerId === session.job.id ? ' is-selected' : ''}`}>
                    <div>
                      {session.job.spec.label} · {session.state}
                    </div>

                    <p className="run-description">
                      {session.job.spec.executor === 'local' ? 'This machine' : 'GPU host (SSH)'}
                      {' · '}
                      {session.job.spec.metadata?.mode === 'hold'
                        ? 'Hold · simulated responses, no model inference'
                        : 'Real · model inference'}
                    </p>
                    <p className="run-description">
                      {session.job.spec.executor === 'local' ? 'Policy identifier' : 'Checkpoint'}:{' '}
                      {session.status?.checkpoint ??
                        String(
                          session.job.spec.metadata?.checkpoint ?? 'Unknown',
                        )}
                      {!session.status?.checkpoint && session.job.spec.executor === 'ssh' && (
                        <span> · requested; loading not confirmed</span>
                      )}
                    </p>

                    {session.unreachable && (
                      <p role="alert">
                        Cannot reach the GPU host: {session.unreachable} The
                        current server state could not be verified.
                      </p>
                    )}

                    {session.read_error && (
                      <p role="alert">
                        Cannot read session files: {session.read_error}
                      </p>
                    )}

                    {session.status?.last_error && (
                      <p role="alert">
                        Prediction error: {session.status.last_error}
                      </p>
                    )}

                    {session.status?.last_refusal && (
                      <p role="status">
                        Last refused connection: {session.status.last_refusal}
                      </p>
                    )}
                    <p className="run-description">
                      SSH tunnel:{' '}
                      {session.job.spec.executor === 'local'
                        ? 'Not required · local server'
                        : session.tunnel ? session.tunnel.status : 'Not opened'}
                    </p>

                    {session.tunnel?.status === 'running' && (
                      <p className="run-description">
                        SSH process is running. Inference link is not verified.
                      </p>
                    )}

                    <div className="run-session-actions">
                      <button type="button" className="run-button" aria-pressed={effectiveServerId === session.job.id}
                        disabled={Boolean(rollout.owner) || rollout.busy || Boolean(validations.active) || validations.busy}
                        onClick={() => selectServer(session.job.id)}>
                        {effectiveServerId === session.job.id ? 'Selected session' : 'Select session'}
                      </button>
                      {policySessionReady(session) && <button type="button" className="run-button run-button-primary"
                        disabled={Boolean(rollout.owner) || rollout.busy || Boolean(validations.active) || validations.busy || Boolean(sessionError)}
                        onClick={() => { selectServer(session.job.id); selectTab('rollout'); }}>
                        Use for rollout
                      </button>}
                      {session.job.spec.executor === 'ssh' && (
                      <button
                        type="button"
                        className="run-button"
                        disabled={
                          openingTunnelId !== null ||
                          !['running', 'queued'].includes(session.job.status) ||
                          Boolean(session.unreachable) ||
                          Boolean(session.read_error) ||
                          ['running', 'queued'].includes(
                            session.tunnel?.status ?? '',
                          )
                        }
                        onClick={() => void handleOpenTunnel(session.job.id)}
                      >
                        {openingTunnelId === session.job.id
                          ? 'Opening tunnel…'
                          : 'Open SSH tunnel'}
                      </button>
                      )}
                      <button
                        type="button"
                        className="run-button"
                        disabled={
                          rolloutServerId === session.job.id || rollout.busy || validations.busy || Boolean(rollout.pollError) || rollout.items === null ||
                          stoppingPolicyId !== null ||
                          !['running', 'queued'].includes(session.job.status)
                        }
                        onClick={() => void handleStopPolicy(session.job.id)}
                      >
                        {stoppingPolicyId === session.job.id
                          ? 'Stopping…'
                          : 'Stop policy server'}
                      </button>
                      <button
                        type="button"
                        className="run-button"
                        disabled={logLoading}
                        aria-expanded={logJobId === session.job.id}
                        onClick={() => void handleViewLogs(session.job.id)}
                      >
                        {logJobId === session.job.id ? 'Hide logs' : 'View logs'}
                      </button>
                    </div>
                    {logJobId === session.job.id && (
                      <div className="run-session-log">
                        {logError && <p role="alert">{logError}</p>}

                        <pre className="run-log">
                          {logLoading
                            ? 'Loading logs…'
                            : logContent || 'No log output yet.'}
                        </pre>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        <div className="run-controls">
          <section className="run-panel">
            <h2>Policy settings</h2>

            <label className="run-field">
              Run on
              <select
                value={target}
                disabled={startingPolicy}
                onChange={(event) => {
                  const next = event.target.value as 'ssh' | 'local';
                  setTarget(next);
                  if (next === 'local') setMode('hold');
                  setPolicyError('');
                }}
              >
                <option value="ssh">GPU host (SSH)</option>
                <option value="local">This machine · hold only</option>
              </select>
            </label>

            <label className="run-field">
              Server mode
              <select
                value={mode}
                disabled={startingPolicy}
                onChange={(event) => setMode(event.target.value as 'real' | 'hold')}
              >
                <option value="real" disabled={target === 'local'}>Real · model inference</option>
                <option value="hold">Hold · connection testing</option>
              </select>
            </label>
            <p className="run-description">
              {mode === 'hold'
                ? 'Hold returns the joint positions sent by the client. It does not load a model or test prediction quality.'
                : 'Real loads the selected checkpoint and uses the GPU to predict actions.'}
            </p>

            {target === 'ssh' ? (
              <>
                <label className="run-field">
                  Checkpoint
                  <select
                    value={checkpoint}
                    disabled={startingPolicy}
                    onChange={(event) => setCheckpoint(event.target.value)}
                  >
                    <option value="">
                      {checkpoints === null
                        ? 'Loading checkpoints…'
                        : 'Select a checkpoint'}
                    </option>
                    {checkpoints?.map((item) => (
                      <option key={item.job_id} value={item.job_id}>
                        {item.label} · step {item.step}
                      </option>
                    ))}
                  </select>
                </label>
                {checkpointError && <p role="alert">{checkpointError}</p>}
              </>
            ) : (
              <>
                <label className="run-field">
                  Recorded dataset
                  <select
                    value={datasetName}
                    disabled={startingPolicy}
                    onChange={(event) => setDatasetName(event.target.value)}
                  >
                    <option value="">
                      {datasets === null
                        ? 'Loading datasets…'
                        : 'Select a recorded dataset'}
                    </option>
                    {datasets?.map((dataset) => (
                      <option key={dataset.id} value={dataset.name}>
                        {dataset.name}
                      </option>
                    ))}
                  </select>
                </label>
                {datasetError && <p role="alert">{datasetError}</p>}
                {datasets?.length === 0 && (
                  <p className="run-description">No recorded datasets available. Fake collections cannot be used.</p>
                )}
                <p className="run-description">
                  The dataset supplies camera and joint information for the test server.
                  No GPU or SSH tunnel is needed. LeRobot with async inference support
                  must be installed in the lab backend environment.
                </p>
              </>
            )}
            <p className="run-description">
              Start robot execution to send live observations, or run the dataset
              client check separately from the command line.
            </p>

            <button
              type="button"
              className="run-button"
              disabled={!canStartPolicy}
              onClick={() => void handleStartPolicy()}
            >
              {startingPolicy ? 'Starting…' : 'Start policy server'}
            </button>

            {policyError && <p role="alert">{policyError}</p>}
          </section>

        </div>
      </div>

      <div className="run-main" role="tabpanel" id="run-panel-rollout" aria-labelledby="run-tab-rollout" hidden={tab !== 'rollout'}>
        <RolloutPanel sessions={sessions} sessionError={sessionError} rollout={rollout} validations={validations} onOpenResults={onOpenResults}
          serverId={effectiveServerId} onServerChange={selectServer} onPrepareServer={() => selectTab('policy')}
          onOpenWorkspace={() => setWorkspaceOpen(true)} />
      </div>
      <RolloutWorkspace rollout={rollout} validations={validations} open={workspaceOpen || Boolean(rollout.owner)} onOpenResults={onOpenResults}
        onClose={() => setWorkspaceOpen(false)} onKeepOpen={() => setWorkspaceOpen(true)} />
    </main>
  );
}

export default RunPage;
