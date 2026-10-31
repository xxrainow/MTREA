import { useCallback, useEffect, useState } from 'react';
import type { DatasetReference } from '../api/datasets';
import {
  findActiveSyncJob,
  loadDatasetRemote,
  loadJob,
  loadRemoteStatus,
  loadSyncProgress,
  syncDataset,
} from '../api/remote';
import type {
  DatasetRemote,
  RemoteState,
  RemoteStatus,
  SyncJob,
  SyncProgress,
} from '../api/remote';

type Props = {
  dataset: DatasetReference;
  onOpenJob?: (jobId: string) => void;
};

const stateLabels: Record<RemoteState, string> = {
  absent: 'Not on GPU server',
  incoming: 'Partially sent',
  synced: 'Synced',
  different: 'Different copy on server',
  unknown: 'Unknown',
};

const phaseLabels: Record<SyncProgress['phase'], string> = {
  hash: 'Fingerprinting',
  transfer: 'Sending',
  verify: 'Verifying on server',
};

const POLL_MS = 1000;

function errorText(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback;
}

type Snapshot = { status: RemoteStatus; remote: DatasetRemote; active: SyncJob | null };

async function fetchSnapshot(dataset: DatasetReference, signal: AbortSignal): Promise<Snapshot> {
  const [status, remote, active] = await Promise.all([
    loadRemoteStatus(signal),
    loadDatasetRemote(dataset, signal),
    findActiveSyncJob(dataset, signal),
  ]);
  return { status, remote, active };
}

function DatasetRemotePanel({ dataset, onOpenJob }: Props) {
  const [status, setStatus] = useState<RemoteStatus | null>(null);
  const [remote, setRemote] = useState<DatasetRemote | null>(null);
  const [job, setJob] = useState<SyncJob | null>(null);
  const [progress, setProgress] = useState<SyncProgress | null>(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(
    (signal: AbortSignal) =>
      fetchSnapshot(dataset, signal)
        .then(({ status: nextStatus, remote: nextRemote, active }) => {
          if (signal.aborted) return;
          setStatus(nextStatus);
          setRemote(nextRemote);
          setJob((current) => active ?? (current?.status === 'running' ? null : current));
          setError('');
        })
        .catch((reason: unknown) => {
          if (!signal.aborted) setError(errorText(reason, 'Cannot load GPU server state.'));
        }),
    [dataset],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const running = job?.status === 'running';
  const jobId = job?.id;

  useEffect(() => {
    if (!running || !jobId) return;
    const controller = new AbortController();
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const [next, nextProgress] = await Promise.all([
            loadJob(jobId, controller.signal),
            loadSyncProgress(jobId, controller.signal),
          ]);
          if (controller.signal.aborted) return;
          setProgress(nextProgress);
          if (next.status !== 'running') setJob(next);
        } catch {
          // A missed poll is retried on the next tick.
        }
      })();
    }, POLL_MS);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [running, jobId]);

  // Its own effect: a finished job ends the polling effect above, which aborts that one's signal.
  const finished = job !== null && !running;

  useEffect(() => {
    if (!finished) return;
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [finished, refresh]);

  async function handleSync() {
    setSubmitting(true);
    setError('');
    setProgress(null);
    try {
      setJob(await syncDataset(dataset));
    } catch (reason: unknown) {
      setError(errorText(reason, 'Cannot start sync.'));
    } finally {
      setSubmitting(false);
    }
  }

  async function copyPath(path: string) {
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setError('Cannot copy to the clipboard; select the path instead.');
    }
  }

  // Not sendable comes first: configuring a host never makes a fake dataset sendable.
  const disabledReason = !status
    ? 'Checking the GPU server…'
    : remote && !remote.sendable
      ? remote.reason ?? 'Not sendable.'
      : !status.configured
        ? `GPU server not configured: ${status.message}`
        : remote?.state === 'synced'
          ? 'Already synced.'
          : remote?.state === 'different'
            ? 'A different copy with this name is on the server. Datasets there are never replaced; use a new name.'
            : '';
  const canSync = !disabledReason && !running && !submitting && remote !== null;

  return (
    <section className="datasets-panel datasets-remote" aria-label="GPU server">
      <div className="datasets-panel-header">
        <h2>GPU server</h2>
        {remote && (
          <span className={`datasets-remote-badge is-${remote.state}`}>
            {stateLabels[remote.state]}
          </span>
        )}
      </div>
      {error && (
        <p className="datasets-error" role="alert">
          {error}
        </p>
      )}
      <dl className="datasets-details">
        <div>
          <dt>Server</dt>
          <dd>
            {status === null
              ? 'Checking…'
              : !status.configured
                ? 'Not configured'
                : `${status.host} · ${status.reachable ? 'reachable' : 'unreachable'}`}
          </dd>
        </div>
        <div>
          <dt>Remote path</dt>
          <dd className="datasets-remote-path">
            {remote?.remote_path ? (
              <>
                <code>{remote.remote_path}</code>
                <button
                  type="button"
                  className="datasets-button"
                  onClick={() => void copyPath(remote.remote_path!)}
                >
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </>
            ) : (
              '—'
            )}
          </dd>
        </div>
        <div>
          <dt>Synced at</dt>
          <dd>{remote?.synced_at ? new Date(remote.synced_at).toLocaleString() : '—'}</dd>
        </div>
      </dl>
      {status?.configured && !status.reachable && (
        <p className="datasets-note">{status.message}</p>
      )}
      {remote?.message && <p className="datasets-note">{remote.message}</p>}
      {running && (
        <div className="datasets-remote-progress" role="status">
          <span>
            {progress
              ? `${phaseLabels[progress.phase]} · ${progress.percent}%`
              : 'Starting…'}
          </span>
          <progress max={100} value={progress?.percent ?? 0} />
        </div>
      )}
      {job && !running && job.status !== 'done' && (
        <p className="datasets-error" role="alert">
          Last sync {job.status}. See the job log for details.
        </p>
      )}
      <div className="datasets-remote-actions">
        <button
          type="button"
          className="datasets-button"
          disabled={!canSync}
          title={disabledReason || undefined}
          onClick={() => void handleSync()}
        >
          {running ? 'Syncing…' : submitting ? 'Starting…' : 'Sync to GPU server'}
        </button>
        {job && onOpenJob && (
          <button
            type="button"
            className="datasets-link-button"
            onClick={() => onOpenJob(job.id)}
          >
            Open job {job.id}
          </button>
        )}
      </div>
      {disabledReason && !running && (
        <p className="datasets-note">{disabledReason}</p>
      )}
    </section>
  );
}

export default DatasetRemotePanel;
