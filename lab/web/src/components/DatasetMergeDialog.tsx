import { useEffect, useRef, useState } from 'react';
import { checkDatasetMerge, loadDataset, loadDatasets, startDatasetMerge } from '../api/datasets';
import type { DatasetSummary, MergeCheck } from '../api/datasets';
import { loadJob } from '../api/remote';
import type { SyncJob } from '../api/remote';
import './DatasetMergeDialog.css';

type Props = {
  dataset: DatasetSummary;
  onClose: () => void;
  onOpenDataset: (dataset: DatasetSummary) => void;
  onOpenJob?: (id: string) => void;
};

function incompatible(source: DatasetSummary, current: DatasetSummary): string {
  if (source.backend !== current.backend) return 'Different mode';
  if (source.issues.length) return 'Metadata needs attention';
  if (!source.episodes) return 'No saved episodes';
  if (source.fps !== current.fps) return 'Different FPS';
  if (source.robot_type !== current.robot_type) return 'Different robot';
  if (source.cameras.join('|') !== current.cameras.join('|')) return 'Different cameras';
  return '';
}

export default function DatasetMergeDialog({ dataset, onClose, onOpenDataset, onOpenJob }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const busyRef = useRef(false);
  const closed = useRef(false);
  const [catalog, setCatalog] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<DatasetSummary[]>([dataset]);
  const [search, setSearch] = useState('');
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const [checked, setChecked] = useState<MergeCheck | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [job, setJob] = useState<SyncJob | null>(null);

  useEffect(() => {
    closed.current = false;
    const element = dialog.current;
    element?.showModal();
    const controller = new AbortController();
    loadDatasets(controller.signal).then((items) => {
      if (!controller.signal.aborted) { setCatalog(items); setLoading(false); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) { setError(String(reason)); setLoading(false); }
    });
    return () => { closed.current = true; controller.abort(); element?.close(); };
  }, []);

  const running = job?.status === 'running' || job?.status === 'queued';
  const jobId = job?.id;
  useEffect(() => {
    if (!running || !jobId) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const next = await loadJob(jobId!, controller.signal);
        if (!controller.signal.aborted) { setJob(next); setError(''); }
      } catch (reason) {
        if (!controller.signal.aborted) setError(`Cannot refresh job status: ${String(reason)}. Retrying…`);
      }
      if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), 2000);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [jobId, running]);

  const candidates = catalog.filter((item) => item.backend === dataset.backend && !selected.some((source) => source.id === item.id) && item.name.toLowerCase().includes(search.toLowerCase()));
  const episodes = selected.reduce((sum, item) => sum + (item.episodes ?? 0), 0);
  const frames = selected.every((item) => item.total_frames !== null) ? selected.reduce((sum, item) => sum + item.total_frames!, 0) : null;
  const nameError = !name ? '' : !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(name) ? 'Use letters, numbers, dots, underscores or hyphens; start with a letter, number or underscore.' : catalog.some((item) => item.backend === dataset.backend && item.name === name) ? 'This name already exists. Choose a new name.' : '';
  const canCheck = !busy && !loading && selected.length >= 2 && !!name && !nameError;

  function closeDialog() {
    closed.current = true;
    dialog.current?.close();
    onClose();
  }

  async function review() {
    if (busyRef.current || !canCheck) return;
    busyRef.current = true; setBusy(true); setError('');
    try {
      const result = await checkDatasetMerge({ backend: dataset.backend, sources: selected.map((item) => item.name), name });
      if (closed.current) return;
      setChecked(result);
      setAdding(false);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot check merge.'); }
    finally { busyRef.current = false; setBusy(false); }
  }

  async function submit() {
    if (!checked || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError('');
    try {
      const result = await startDatasetMerge(checked);
      if (!closed.current) setJob(result.job);
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot start merge.'); setChecked(null); }
    finally { busyRef.current = false; setBusy(false); }
  }

  async function openOutput() {
    if (!checked || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError('');
    try {
      const result = await loadDataset(checked, new AbortController().signal);
      if (closed.current) return;
      dialog.current?.close();
      onOpenDataset(result.dataset);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot open output.'); }
    finally { busyRef.current = false; setBusy(false); }
  }

  const title = job ? (running ? 'Merging datasets' : job.status === 'done' ? 'Merge complete' : `Merge ${job.status}`) : checked ? 'Review merge' : 'Merge datasets';
  return <dialog ref={dialog} className="merge-dialog" aria-labelledby="merge-title" onCancel={(event) => { event.preventDefault(); closeDialog(); }}>
    <div className="merge-dialog-heading"><div><span className="merge-eyebrow">LOCAL DATASETS / {dataset.backend.toUpperCase()}</span><h2 id="merge-title">{title}</h2></div><button className="datasets-button" aria-label="Close merge" onClick={closeDialog}>×</button></div>
    {error && <p className="datasets-error" role="alert">{error}</p>}
    {job ? <>
      <div className="merge-review-output"><h3>{checked?.name}</h3><p role="status">{running ? 'Combining and verifying saved episodes…' : job.status === 'done' ? `${checked?.episodes} episodes merged into a new local dataset.` : 'The job ended without confirmed completion. Check the job log for details.'}</p>{running && <progress aria-label="Merging datasets" />}<code>{checked?.location}</code></div>
      <p className="merge-description">Source datasets are preserved. You can close this dialog and follow the job from Jobs.{dataset.backend === 'fake' && ' This fake output contains metadata only.'}</p>
      <footer>{onOpenJob && <button className="datasets-button" onClick={() => onOpenJob(job.id)}>View job</button>}<button className="datasets-button" onClick={closeDialog}>Close</button>{job.status === 'done' && <button className="datasets-button merge-primary" disabled={busy} onClick={() => void openOutput()}>View dataset</button>}</footer>
    </> : checked ? <>
      <p className="merge-description">Recording settings and metadata checks passed. The output will be verified before it appears in Datasets.</p>
      <div className="merge-review-output"><span className="merge-eyebrow">NEW DATASET</span><h3>{checked.name}</h3><p>{checked.episodes} episodes · {checked.total_frames?.toLocaleString() ?? '—'} frames</p><code>{checked.location}</code></div>
      <h3 className="merge-review-label">Included sources</h3>{selected.map((item) => <div className="merge-review-row" key={item.id}><span>{item.name}</span><span>{item.episodes} episodes</span></div>)}
      <p className="merge-description">Creates a separate local dataset and preserves the originals. Research roles and demonstration quality are not validated by these checks.</p>
      <footer><button className="datasets-button" disabled={busy} onClick={() => setChecked(null)}>← Back</button><button className="datasets-button merge-primary" disabled={busy} onClick={() => void submit()}>{busy ? 'Starting…' : 'Merge datasets'}</button></footer>
    </> : <>
      <p className="merge-description">Combine saved episodes into a new local dataset. Source datasets stay unchanged.{dataset.backend === 'fake' && ' Fake datasets contain metadata only.'}</p>
      <div className="merge-section-label"><h3>Source datasets</h3><span>{selected.length} selected</span></div>
      <div className="merge-source-list">{selected.map((item) => <div className="merge-source" key={item.id}><span className="merge-source-icon">▤</span><div><strong>{item.name}</strong><small>{item.episodes ?? '—'} episodes · {item.total_frames?.toLocaleString() ?? '—'} frames · {item.backend}</small></div>{item.id === dataset.id ? <span className="merge-chip">Current</span> : <button className="merge-remove" disabled={busy} aria-label={`Remove ${item.name}`} onClick={() => setSelected(selected.filter((source) => source.id !== item.id))}>×</button>}</div>)}</div>
      <button className="datasets-button merge-add" disabled={busy || loading || selected.length >= 50} aria-expanded={adding} onClick={() => setAdding(!adding)}>{loading ? 'Loading…' : '+ Add dataset'}</button>
      {adding && <div className="merge-picker"><input autoFocus aria-label="Search datasets" placeholder="Search local datasets…" value={search} onChange={(event) => setSearch(event.target.value)} /><div className="merge-picker-results">{candidates.map((item) => { const reason = incompatible(item, dataset); return <button key={item.id} disabled={busy || !!reason} onClick={() => { setSelected([...selected, item]); setAdding(false); setSearch(''); }}><span>{item.name}<small>{item.episodes ?? '—'} episodes · {reason || 'Available for compatibility check'}</small></span><span>{reason ? 'Unavailable' : '+'}</span></button>; })}{!candidates.length && <p>No matching datasets.</p>}</div><small>Only {dataset.backend} datasets are shown. Full compatibility is checked before merging.</small></div>}
      {selected.length < 2 && <p className="merge-hint">Add at least one more dataset to continue.</p>}
      <label className="merge-name">Merged dataset name<input value={name} disabled={busy} onChange={(event) => setName(event.target.value)} placeholder="e.g. pickplace_blocks_combined" aria-describedby="merge-name-help" aria-invalid={!!nameError} /></label>
      <p id="merge-name-help" className={nameError ? 'merge-error' : 'merge-hint'}>{nameError || `New local dataset: data/${dataset.backend === 'fake' ? 'fake/collection/' : ''}raw/${name || '<name>'}`}</p>
      <div className="merge-total"><div><strong>{episodes}</strong><span>Total episodes</span></div><div><strong>{frames?.toLocaleString() ?? '—'}</strong><span>Total frames</span></div><div><strong>{selected.length}</strong><span>Source datasets</span></div></div>
      <p className="merge-hint">Keep oracle and held-out demos out of training datasets for compared methods. Merging does not assign research roles.</p>
      <footer><button className="datasets-button" onClick={closeDialog}>Cancel</button><button className="datasets-button merge-primary" disabled={!canCheck} onClick={() => void review()}>{busy ? 'Checking…' : 'Review merge →'}</button></footer>
    </>}
  </dialog>;
}
