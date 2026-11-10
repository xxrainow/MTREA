import { useEffect, useState } from 'react';
import { loadDatasets } from '../api/datasets';
import type { DatasetSummary } from '../api/datasets';
import { errorMessage } from '../api/hardware';
import {
  loadPolicyCheckpoints,
  loadPolicySessions,
  openPolicyTunnel,
  startPolicy,
} from '../api/serving';
import type { PolicyCheckpoint, PolicySession } from '../api/serving';
import { loadJobLog, stopJob } from '../api/training';

import RolloutPanel from '../components/RolloutPanel';

import './RunPage.css';

function RunPage() {
  const [target, setTarget] = useState<'ssh' | 'local'>('ssh');
  const [mode, setMode] = useState<'real' | 'hold'>('real');
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [checkpoint, setCheckpoint] = useState('');
  const [rolloutServerId, setRolloutServerId] = useState<string | null>(null);
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
    !startingPolicy;

  async function handleStartPolicy() {
    if (!canStartPolicy) return;

    setStartingPolicy(true);
    setPolicyError('');

    try {
      await startPolicy({
        target,
        mode,
        source_job_id: target === 'ssh' ? checkpoint : null,
        dataset_name: target === 'local' ? datasetName : null,
      });
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setStartingPolicy(false);
    }
  }

  async function handleStopPolicy(jobId: string) {
    if (stoppingPolicyId !== null || rolloutServerId === jobId) return;

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

      <div className="run-layout">
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
                  <li key={session.job.id} className="run-session">
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
                          rolloutServerId === session.job.id ||
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
          <RolloutPanel sessions={sessions} sessionError={sessionError} onServerInUse={setRolloutServerId} />
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

          <section className="run-panel">
            <h2>Rollout results</h2>

            <p className="run-description">
              Record whether the rollout succeeded after execution finishes.
            </p>

            <div className="run-actions">
              <button type="button" className="run-button" disabled>
                Success
              </button>
              <button type="button" className="run-button" disabled>
                Failure
              </button>
            </div>

            <p className="run-note">
              Result recording integration is pending. Results will be appended
              without deleting or overwriting existing records.
            </p>
          </section>
        </div>
      </div>

      <section className="run-panel run-history">
        <div className="run-panel-header">
          <h2>Evaluation history</h2>
          <span className="run-status">Integration pending</span>
        </div>

        <p className="run-description">
          Rollout records and absolute success rates for each task will be
          displayed here.
        </p>
      </section>
    </main>
  );
}

export default RunPage;
