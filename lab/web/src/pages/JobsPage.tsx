import { Fragment, useEffect, useState } from 'react';
import './JobsPage.css';
import type { TrainingRun } from '../training/model';
import TrainingMetricChart from '../components/TrainingMetricChart';

type Job = {
  id: string;
  spec: {
    label: string;
    kind: string;
    executor: string;
  };
  status: string;
  pid: number | null;
  submitted_at: string;
};

const statusLabels: Record<string, string> = {
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  stopped: 'Stopped',
  unknown: 'Unknown',
};

function JobsPage({ trainingRuns = [] }: { trainingRuns?: TrainingRun[] }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [stoppingJobId, setStoppingJobId] = useState<string | null>(null);

  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [logContent, setLogContent] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logError, setLogError] = useState<string | null>(null);
  const [logRefreshKey, setLogRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    async function loadJobs() {
      try {
        const response = await fetch('/api/jobs', {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch jobs: HTTP ${response.status}`);
        }

        const data: Job[] = await response.json();
        setJobs(data);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        setError(
          error instanceof Error ? error.message : 'Unable to load jobs.',
        );
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadJobs();

    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    if (selectedJobId === null) {
      return;
    }

    const controller = new AbortController();

    async function loadLog() {
      setLogLoading(true);
      setLogError(null);
      setLogContent('');

      try {
        const response = await fetch(
          `/api/jobs/${encodeURIComponent(selectedJobId!)}` +
            '/logs?tail_lines=100',
          { signal: controller.signal },
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch logs: HTTP ${response.status}`);
        }

        const data: { job_id: string; content: string } = await response.json();

        if (!controller.signal.aborted) {
          setLogContent(data.content);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        setLogError(
          error instanceof Error ? error.message : 'Unable to load logs.',
        );
      } finally {
        if (!controller.signal.aborted) {
          setLogLoading(false);
        }
      }
    }

    void loadLog();

    return () => controller.abort();
  }, [selectedJobId, logRefreshKey]);

  function handleViewLog(jobId: string) {
    setLogLoading(true);
    setLogError(null);
    setLogContent('');
    setSelectedJobId(jobId);
    setLogRefreshKey((previous) => previous + 1);
  }

  function handleRefresh() {
    setLoading(true);
    setError(null);
    setRefreshKey((previous) => previous + 1);
  }

  async function handleStop(jobId: string) {
    setStoppingJobId(jobId);
    setError(null);

    try {
      const response = await fetch(
        `/api/jobs/${encodeURIComponent(jobId)}/stop`,
        { method: 'POST' },
      );

      if (!response.ok) {
        throw new Error(`Failed to stop job: HTTP ${response.status}`);
      }

      handleRefresh();
    } catch (error) {
      setError(
        error instanceof Error ? error.message : 'Unable to stop the job.',
      );
    } finally {
      setStoppingJobId(null);
    }
  }

  return (
    <main className="jobs-page">
      <div className="jobs-heading">
        <div>
          <h1>Jobs</h1>
          <p>Check the job status and perform log retrieval and termination.</p>
        </div>

        <button
          type="button"
          className="jobs-button"
          onClick={handleRefresh}
          disabled={loading || stoppingJobId !== null}
        >
          {loading ? 'loading...' : 'Refresh'}
        </button>
      </div>

      <section className="jobs-panel" aria-label="Simulated training jobs">
        <h2>Training jobs · Simulation</h2>
        <p className="jobs-note">One job per training run. These browser-only records are separate from backend jobs and disappear on reload.</p>
        {!trainingRuns.length && <p>No simulated training jobs.</p>}
        {[...trainingRuns].reverse().map(run => <details key={run.id}>
          <summary>{run.config.runName} · {run.status} · {run.step.toLocaleString()} / {Number(run.config.steps).toLocaleString()} steps</summary>
          <p>{run.id} · {run.config.target} · Simulated</p>
          <div className="train-grid"><TrainingMetricChart metrics={run.metrics} kind="loss" /><TrainingMetricChart metrics={run.metrics} kind="lr" /></div>
          <pre className="log-output">{run.logs.join('\n')}</pre>
        </details>)}
      </section>

      <section className="jobs-panel" aria-label="Job list">
        <div className="jobs-panel-header">
          <h2>Job list</h2>
          <span className="jobs-count">
            {loading ? 'loading' : `${jobs.length}`}
          </span>
        </div>

        {loading && <p role="status">Loading jobs</p>}

        {error && <p role="alert">{error}</p>}

        {!loading && !error && jobs.length === 0 && (
          <p className="jobs-empty">No jobs</p>
        )}

        {!loading && !error && jobs.length > 0 && (
          <div className="jobs-table-wrapper">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th scope="col">Job name</th>
                  <th scope="col">Spec</th>
                  <th scope="col">Executor</th>
                  <th scope="col">Status</th>
                  <th scope="col">PID</th>
                  <th scope="col">Started time</th>
                  <th scope="col">Configuration</th>
                </tr>
              </thead>

              <tbody>
                {jobs.map((job) => (
                  <Fragment key={job.id}>
                    <tr>
                      <td>
                        <span className="jobs-name">
                          {job.spec.label || job.id}
                        </span>
                        {job.spec.label && (
                          <span className="jobs-id">{job.id}</span>
                        )}
                      </td>

                      <td>{job.spec.kind}</td>
                      <td>{job.spec.executor}</td>

                      <td>
                        <span
                          className={`jobs-status jobs-status-${job.status}`}
                        >
                          {statusLabels[job.status] || job.status}
                        </span>
                      </td>

                      <td>{job.pid ?? '—'}</td>
                      <td>{new Date(job.submitted_at).toLocaleString('en-US')}</td>

                      <td>
                        <div className="jobs-actions">
                          <button
                            type="button"
                            className="jobs-button"
                            aria-expanded={selectedJobId === job.id}
                            onClick={() => {
                              if (selectedJobId === job.id) {
                                setSelectedJobId(null);
                              } else {
                                handleViewLog(job.id);
                              }
                            }}
                          >
                            {selectedJobId === job.id ? 'Log close' : 'Log'}
                          </button>

                          <button
                            type="button"
                            className="jobs-button"
                            onClick={() => void handleStop(job.id)}
                            disabled={
                              job.status !== 'running' ||
                              stoppingJobId !== null ||
                              loading
                            }
                          >
                            {stoppingJobId === job.id ? 'stopping' : 'stopped'}
                          </button>
                        </div>
                      </td>
                    </tr>

                    {selectedJobId === job.id && (
                      <tr className="jobs-log-row">
                        <td colSpan={7}>
                          <section
                            className="jobs-inline-log"
                            aria-label={`${job.spec.label || job.id} log`}
                          >
                            <div className="jobs-panel-header">
                              <div>
                                <h3>Logs</h3>
                                <p className="jobs-log-description">
                                  Last 100 lines
                                </p>
                              </div>

                              <button
                                type="button"
                                className="jobs-button"
                                onClick={() => handleViewLog(job.id)}
                                disabled={logLoading}
                              >
                                {logLoading ? 'Loading' : 'Log Refresh'}
                              </button>
                            </div>

                            {logLoading && <p role="status">Loading logs</p>}

                            {logError && <p role="alert">{logError}</p>}

                            {!logLoading &&
                              !logError &&
                              (logContent.length > 0 ? (
                                <pre className="log-output">{logContent}</pre>
                              ) : (
                                <p className="jobs-empty">No logs.</p>
                              ))}
                          </section>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="jobs-note">
          The result is "Unknown" because the process has been terminated, but
          the exit code has not been obtained.
        </p>
      </section>
    </main>
  );
}

export default JobsPage;
