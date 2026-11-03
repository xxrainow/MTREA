import { useEffect, useRef, useState } from 'react';
import { checkDatasetEdit, loadDataset, startDatasetEdit } from '../api/datasets';
import type { DatasetDetail, DatasetEditCheck, DatasetSummary } from '../api/datasets';
import { loadJob } from '../api/remote';
import type { SyncJob } from '../api/remote';
import './DatasetMergeDialog.css';
import './DatasetEditDialog.css';

type Props = {
  detail: DatasetDetail;
  onClose: () => void;
  onOpenDataset: (dataset: DatasetSummary) => void;
  onOpenJob?: (id: string) => void;
};

export default function DatasetEditDialog({ detail, onClose, onOpenDataset, onOpenJob }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const closed = useRef(false);
  const inFlight = useRef(false);
  const { dataset, tasks } = detail;
  const [name, setName] = useState(`${dataset.name}_edited`);
  const [instructions, setInstructions] = useState(() => Object.fromEntries(tasks.map((task) => [task, task])));
  const [checked, setChecked] = useState<DatasetEditCheck | null>(null);
  const [job, setJob] = useState<SyncJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const element = dialog.current;
    closed.current = false;
    element?.showModal();
    return () => { closed.current = true; element?.close(); };
  }, []);

  function close() {
    closed.current = true;
    dialog.current?.close();
    onClose();
  }

  const running = job?.status === 'queued' || job?.status === 'running';
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

  async function request(action: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError('');
    try { await action(); }
    catch (reason) {
      if (!closed.current) setError(reason instanceof Error ? reason.message : 'Cannot save dataset edits.');
    } finally {
      inFlight.current = false;
      if (!closed.current) setBusy(false);
    }
  }

  function review() {
    void request(async () => {
      const result = await checkDatasetEdit(dataset, { name, instructions });
      if (!closed.current) setChecked(result);
    });
  }

  function save() {
    if (!checked) return;
    void request(async () => {
      const result = await startDatasetEdit(dataset, checked);
      if (!closed.current) setJob(result.job);
    });
  }

  function openOutput() {
    if (!checked) return;
    void request(async () => {
      const result = await loadDataset({ backend: checked.backend, name: checked.name }, new AbortController().signal);
      if (closed.current) return;
      dialog.current?.close();
      onOpenDataset(result.dataset);
    });
  }

  const nameError = name === dataset.name ? 'Choose a new name to preserve the original.'
    : !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(name) ? 'Use letters, numbers, dots, underscores or hyphens; start with a letter, number or underscore.' : '';
  const values = Object.values(instructions).map((value) => value.trim());
  const instructionError = values.some((value) => !value) ? 'Instructions cannot be empty.'
    : new Set(values).size !== values.length ? 'Each instruction must remain distinct.' : '';
  const title = job ? (running ? 'Saving edited copy' : job.status === 'done' ? 'Edited copy saved' : `Save ${job.status}`)
    : checked ? 'Review edited copy' : 'Edit dataset';

  return <dialog ref={dialog} className="merge-dialog dataset-edit-dialog" aria-labelledby="edit-title"
    onCancel={(event) => { event.preventDefault(); close(); }}>
    <div className="merge-dialog-heading">
      <div><span className="merge-eyebrow">NEW COPY / {dataset.backend.toUpperCase()}</span><h2 id="edit-title">{title}</h2></div>
      <button type="button" className="datasets-button" aria-label="Close dataset edit" onClick={close}>×</button>
    </div>
    {error && <p className="datasets-error" role="alert">{error}</p>}
    {job ? <>
      <div className="merge-review-output">
        <h3>{checked?.name}</h3>
        <p role="status">{running ? 'Copying recordings and updating instruction metadata…'
          : job.status === 'done' ? `${checked?.episodes} episodes saved with the edited instructions.`
            : 'Saving ended without confirmed completion. See the job log for details.'}</p>
        {running && <progress aria-label="Saving edited copy" />}
        <code>{checked?.location}</code>
      </div>
      <p className="merge-description">The original dataset is unchanged. This job remains available in Jobs after closing.</p>
      <footer>
        {onOpenJob && <button className="datasets-button" onClick={() => onOpenJob(job.id)}>View job</button>}
        <button className="datasets-button" onClick={close}>Close</button>
        {job.status === 'done' && <button className="datasets-button merge-primary" disabled={busy} onClick={openOutput}>View dataset</button>}
      </footer>
    </> : checked ? <>
      <div className="merge-review-output">
        <span className="merge-eyebrow">NEW DATASET</span><h3>{checked.name}</h3>
        <p>{checked.episodes} episodes · from {dataset.name}</p><code>{checked.location}</code>
      </div>
      <h3 className="merge-review-label">Instructions in the new copy</h3>
      {tasks.map((task) => <div className="edit-review-instruction" key={task}>
        {task !== checked.instructions[task] && <small>Previously: {task}</small>}
        <p>{checked.instructions[task]}</p>
      </div>)}
      <p className="merge-description">Recordings and task IDs are preserved. The original, its collection history and any GPU copy stay unchanged. Sync the new dataset separately when needed.</p>
      <footer>
        <button className="datasets-button" disabled={busy} onClick={() => setChecked(null)}>← Back</button>
        <button className="datasets-button merge-primary" disabled={busy} onClick={save}>{busy ? 'Starting…' : 'Save new copy'}</button>
      </footer>
    </> : <>
      <p className="merge-description">Change the dataset name and instructions, then save a separate copy. Original: <strong>{dataset.name}</strong></p>
      <label className="merge-name">Dataset name
        <input value={name} disabled={busy} onChange={(event) => setName(event.target.value)} aria-invalid={!!nameError} aria-describedby="edit-name-help" />
      </label>
      <p id="edit-name-help" className={nameError ? 'merge-error' : 'merge-hint'}>{nameError || 'A new name is required. The original dataset is kept.'}</p>
      {tasks.map((task, index) => <label className="edit-instruction" key={task}>
        <span>{tasks.length === 1 ? 'Instruction' : `Instruction ${index + 1}`}</span>
        <textarea value={instructions[task]} maxLength={2000} rows={3} disabled={busy}
          onChange={(event) => setInstructions({ ...instructions, [task]: event.target.value })} />
        <small>Applies to {detail.episodes.filter((ep) => ep.tasks.includes(task)).length} episodes in the new copy.</small>
      </label>)}
      {instructionError && <p className="merge-error" role="alert">{instructionError}</p>}
      <p className="merge-hint">Instruction edits update the saved task metadata used for training. Research task roles are not changed.{dataset.backend === 'fake' && ' Fake datasets contain metadata only.'}</p>
      <footer>
        <button className="datasets-button" onClick={close}>Cancel</button>
        <button className="datasets-button merge-primary" disabled={busy || !!nameError || !!instructionError || !tasks.length} onClick={review}>{busy ? 'Checking…' : 'Review changes →'}</button>
      </footer>
    </>}
  </dialog>;
}
