import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { mergeResults, token } from '../api/rolloutValidations';
import type { MergedResult } from '../api/rolloutValidations';
import type { ResultRun } from '../rollouts/results';
import { formatResultDate } from '../rollouts/results';
import './DatasetMergeDialog.css';

export default function RolloutMergeDialog({ current, candidates, onClose, onMerged }: {
  current: ResultRun; candidates: ResultRun[]; onClose: () => void; onMerged: (value: MergedResult) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const closed = useRef(false);
  const submitting = useRef(false);
  const requestId = useRef(token());
  const [selected, setSelected] = useState([current.key]);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    closed.current = false;
    const element = dialog.current;
    element?.showModal();
    return () => { closed.current = true; element?.close(); };
  }, []);
  function close() { closed.current = true; dialog.current?.close(); onClose(); }
  async function submit() {
    if (submitting.current) return;
    submitting.current = true; setBusy(true); setError('');
    try {
      const value = await mergeResults(candidates.filter(item => selected.includes(item.key)).map(item => ({
        kind: item.kind === 'merged' ? 'merged' : item.validation ? 'validation' : 'standalone', id: item.id,
      })), name, requestId.current);
      if (!closed.current) { dialog.current?.close(); onMerged(value); }
    } catch (reason) {
      if (!closed.current) setError(errorMessage(reason));
    } finally {
      submitting.current = false;
      if (!closed.current) setBusy(false);
    }
  }
  return <dialog ref={dialog} className="merge-dialog" aria-labelledby="rollout-merge-title"
    onCancel={event => { event.preventDefault(); close(); }}>
    <div className="merge-dialog-heading"><h2 id="rollout-merge-title">Merge rollouts</h2>
      <button type="button" className="run-button" aria-label="Close merge" onClick={close}>×</button></div>
    <p className="merge-description">Create a new result bundle from runs with the same checkpoint, instruction and recorded execution settings. Originals are preserved. Outcomes and notes are captured as they are now.</p>
    <label className="merge-name">Result name<input value={name} maxLength={120} disabled={busy} onChange={event => setName(event.target.value)} /></label>
    <h3>Source rollouts · {selected.length} selected</h3>
    <div className="merge-picker-results">{candidates.map(item => <button type="button" key={item.key} disabled={busy || item.key === current.key}
      aria-pressed={selected.includes(item.key)} onClick={() => setSelected(items => items.includes(item.key) ? items.filter(key => key !== item.key) : [...items, item.key])}>
      <span>{selected.includes(item.key) ? '✓ ' : '+ '}{item.name ?? item.instruction}<small>{formatResultDate(item.createdAt)} · {item.id}</small></span>
      <span>{item.attempts.length} attempts</span>
    </button>)}</div>
    <p className="merge-description">Selected attempts: {candidates.filter(item => selected.includes(item.key)).reduce((sum, item) => sum + item.attempts.length, 0)}. Overlapping attempts and incompatible settings are refused.</p>
    {error && <p role="alert" className="merge-error">{error}</p>}
    <footer><button type="button" className="run-button" onClick={close}>Cancel</button>
      <button type="button" className="run-button run-button-primary" disabled={busy || selected.length < 2 || !name.trim()} onClick={() => void submit()}>{busy ? 'Merging…' : 'Merge rollouts'}</button></footer>
  </dialog>;
}
