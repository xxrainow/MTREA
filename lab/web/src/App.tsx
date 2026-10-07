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

      <button type="button" onClick={handleRefresh} disabled={loading || stoppingJobId !== null}>
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
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}

export default App;
