import { useEffect, useState } from 'react';
import './App.css';

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

function App() {
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
      <h1>MTREA Lab</h1>
      <h2>Jobs</h2>

      <button
        type="button"
        onClick={handleRefresh}
        disabled={loading || stoppingJobId !== null}
      >
        {loading ? 'Loading…' : 'Refresh'}
      </button>

      {loading && <p role="status">Loading jobs…</p>}

      {error && <p role="alert">{error}</p>}

      {!loading && !error && jobs.length === 0 && <p>No jobs found.</p>}

      {!loading && jobs.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Job name</th>
              <th>Type</th>
              <th>Executor</th>
              <th>Status</th>
              <th>PID</th>
              <th>Submitted at</th>
              <th>Actions</th>
            </tr>
          </thead>

          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td>{job.spec.label || job.id}</td>
                <td>{job.spec.kind}</td>
                <td>{job.spec.executor}</td>
                <td>{job.status}</td>
                <td>{job.pid ?? '—'}</td>
                <td>{new Date(job.submitted_at).toLocaleString('en-US')}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => void handleStop(job.id)}
                    disabled={
                      job.status !== 'running' ||
                      stoppingJobId !== null ||
                      loading
                    }
                  >
                    {stoppingJobId === job.id ? 'Stopping…' : 'Stop'}
                  </button>
                  <button type="button" onClick={() => handleViewLog(job.id)}>
                    View logs
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedJobId !== null && (
        <section className="job-logs" aria-label="Job logs">
          <h2>Logs</h2>
          <p>Job ID: {selectedJobId}</p>
          <p>Last 100 lines</p>

          <button
            type="button"
            onClick={() => handleViewLog(selectedJobId)}
            disabled={logLoading}
          >
            {logLoading ? 'Loading…' : 'Refresh logs'}
          </button>

          <button type="button" onClick={() => setSelectedJobId(null)}>
            Close
          </button>

          {logLoading && <p role="status">Loading logs…</p>}

          {logError && <p role="alert">{logError}</p>}

          {!logLoading &&
            !logError &&
            (logContent.length > 0 ? (
              <pre className="log-output">{logContent}</pre>
            ) : (
              <p>No log output available.</p>
            ))}
        </section>
      )}
    </main>
  );
}

export default App;
