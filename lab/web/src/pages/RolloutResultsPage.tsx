import { useEffect, useState } from 'react';
import { rolloutOwnsDevices } from '../api/rollouts';
import { validationStateLabels } from '../api/rolloutValidations';
import { useRolloutResults } from '../hooks/useRolloutResults';
import { formatResultDate, listRolloutResults, outcomeCounts, resultTargetSelection } from '../rollouts/results';
import type { ResultRun, ResultSelection, RolloutResultTarget } from '../rollouts/results';
import RolloutMergeDialog from '../components/RolloutMergeDialog';
import RolloutResultDetail from '../components/RolloutResultDetail';
import './RunPage.css';
import './RolloutResultsPage.css';

const selectionKey = 'lab.rollout.results.selection';

function savedSelection(): ResultSelection {
  try {
    const value = JSON.parse(sessionStorage.getItem(selectionKey) ?? '{}');
    return value && typeof value === 'object' ? value : {};
  } catch { return {}; }
}

export default function RolloutResultsPage({ initialTarget, onEditingChange, onOpenExecution }: {
  initialTarget: RolloutResultTarget | null;
  onEditingChange: (editing: boolean) => void;
  onOpenExecution: (target: RolloutResultTarget) => void;
}) {
  const results = useRolloutResults();
  const [selection, setSelection] = useState<ResultSelection>(() => initialTarget ? resultTargetSelection(initialTarget) : savedSelection());
  const [search, setSearch] = useState('');
  const [showDeleted, setShowDeleted] = useState(false);
  const [merging, setMerging] = useState(false);
  const [notice, setNotice] = useState('');
  const [dirtyReview, setDirtyReview] = useState({ id: '', dirty: false });
  const runs = results.data ? listRolloutResults(results.data.validations, results.data.trials, results.data.merged) : [];
  const run = runs.find(item => item.key === selection.runKey)
    ?? (selection.runKey && selection.jobId ? runs.find(item => item.attempts.some(trial => trial.rollout.job.id === selection.jobId)) : undefined);
  const attempt = run?.attempts.find(item => item.rollout.job.id === selection.jobId) ?? run?.attempts[0];
  const dirty = dirtyReview.id === attempt?.rollout.job.id && dirtyReview.dirty;
  const blocked = dirty || results.saving || results.archiving || merging;
  const archived = new Set(results.data?.archived.map(item => `${item.kind}:${item.id}`) ?? []);
  const deleted = run ? archived.has(run.key) : false;
  const visibleRuns = runs.filter(item => showDeleted || !archived.has(item.key));
  const counts = run ? outcomeCounts(run.attempts) : null;
  const query = search.toLocaleLowerCase().trim();
  const matchingRuns = visibleRuns.filter(item => [item.name ?? '', item.instruction, item.checkpoint, item.id]
    .some(value => value.toLocaleLowerCase().includes(query)));

  useEffect(() => {
    onEditingChange(blocked);
    return () => onEditingChange(false);
  }, [blocked, onEditingChange]);

  useEffect(() => {
    sessionStorage.setItem(selectionKey, JSON.stringify(selection));
  }, [selection]);

  function choose(next: ResultSelection) {
    if (blocked) return;
    setNotice('');
    setSelection(next);
  }

  function canDelete(item: ResultRun) {
    return (!item.validation || ['completed', 'ended'].includes(item.validation.state))
      && !item.attempts.some(value => rolloutOwnsDevices(value.rollout) || value.rollout.read_error);
  }

  async function changeArchive(item: ResultRun, restore = false) {
    if (blocked || results.error || (!restore && !canDelete(item))) return;
    if (!restore && !window.confirm(`Delete this rollout from results (${item.attempts.length} saved attempts)? Original records are preserved. Use Show deleted to restore it.`)) return;
    const changed = await results.archive({ kind: item.kind === 'merged' ? 'merged' : item.validation ? 'validation' : 'standalone', id: item.id }, restore);
    if (changed) {
      setNotice(restore ? 'Rollout restored.' : 'Rollout removed from results. Original records are preserved.');
      if (!restore) setSelection({});
    }
  }

  return <main className="run-page rollout-results-page">
    {run && <button type="button" className="results-back-button" disabled={blocked} onClick={() => choose({})}>← Rollout list</button>}
    <div className="run-heading results-heading">
      <div><h1>{run ? run.name ?? run.instruction : 'Rollout results'}</h1>
        <p>{run ? `${formatResultDate(run.createdAt)} · ${run.id}` : 'Saved rollout runs and their validation attempts.'}</p></div>
      <div className="run-actions results-heading-actions">
        {run && <><span className="run-status">{deleted ? 'Deleted' : run.validation ? validationStateLabels[run.validation.state] : run.state}</span>
          {!deleted && <button type="button" className="run-button run-button-primary" disabled={blocked || Boolean(results.error) || !canDelete(run) || run.attempts.length === 0}
            onClick={() => setMerging(true)}>Merge</button>}
          <button type="button" className={`run-button ${deleted ? '' : 'results-delete-button'}`}
            disabled={blocked || Boolean(results.error) || (!deleted && !canDelete(run))}
            title={!deleted && !canDelete(run) ? 'End validation and Release the arm first.' : undefined}
            onClick={() => void changeArchive(run, deleted)}>{deleted ? 'Restore' : 'Delete'}</button></>}
      </div>
    </div>
    {merging && run && <RolloutMergeDialog current={run} candidates={runs.filter(item => !archived.has(item.key) && canDelete(item) && item.attempts.length > 0)}
      onClose={() => setMerging(false)} onMerged={value => {
        setMerging(false); setNotice('Merged result saved. Original records are preserved.');
        setSelection({ runKey: `merged:${value.id}` }); results.refresh();
      }} />}
    {results.error && <p role="alert">Cannot refresh results: {results.error}{results.data ? '. Last loaded records are shown; editing is disabled.' : ''}</p>}
    {results.archiveError && <p role="alert">Cannot update result list: {results.archiveError}</p>}
    {notice && <p className="run-note" role="status">{notice}</p>}
    {dirty && <p className="run-note" role="status">Save or discard outcome edits before switching records or leaving this page.</p>}
    {deleted && <p className="run-note">This rollout was deleted from the results list. Original records are preserved. Restore it to access result actions.</p>}
    {selection.runKey && results.data && !run && <p role="status">The requested result is unavailable. Select a saved rollout below.</p>}
    {results.data === null ? <p>{results.error ? 'Retrying automatically…' : 'Loading rollout results…'}</p>
      : <div className="results-content">
        {!run ? <section className="run-panel" aria-label="Rollout runs">
          <div className="run-panel-header"><h2>Rollouts</h2><span className="run-status">{visibleRuns.length} rollouts</span></div>
          <div className="results-list-controls">
            <label className="run-field results-search">Search rollouts
              <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Instruction, checkpoint or rollout ID" />
            </label>
            <label className="results-show-deleted"><input type="checkbox" checked={showDeleted} disabled={blocked}
              onChange={event => setShowDeleted(event.target.checked)} />Show deleted</label>
          </div>
          <div className="results-table-wrapper"><table className="results-rollout-table">
            <thead><tr><th>Recorded</th><th>Instruction</th><th>Checkpoint / policy</th><th>Attempts</th><th>Status</th><th>Actions</th></tr></thead>
            <tbody>{matchingRuns.map(item => <tr key={item.key} className="results-clickable-row"
              tabIndex={blocked ? -1 : 0} aria-label={`Open rollout ${item.id}`}
              onClick={() => choose({ runKey: item.key })}
              onKeyDown={event => {
                if (event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) {
                  event.preventDefault(); choose({ runKey: item.key });
                }
              }}>
              <td><button type="button" className="results-name-button" disabled={blocked} aria-label={`Open rollout details ${item.id}`}
                onClick={event => { event.stopPropagation(); choose({ runKey: item.key }); }}>{formatResultDate(item.createdAt)}</button>
                <small title={item.id}>Rollout {item.id}</small></td>
              <td className="results-instruction-text">{(item.name ?? item.instruction) || '(Empty saved instruction)'}</td>
              <td>{item.checkpoint}<small>{item.mode === 'hold' ? 'Hold · connection test' : item.mode === 'real' ? 'Real inference' : 'Mode not recorded'}</small></td>
              <td className="results-count">{item.attempts.length} / {item.targetCount}<small>recorded / requested</small></td>
              <td><span className="run-status">{archived.has(item.key) ? 'Deleted' : item.validation ? validationStateLabels[item.validation.state] : item.state}</span></td>
              <td><button type="button" className={`run-button ${archived.has(item.key) ? '' : 'results-delete-button'}`}
                disabled={blocked || Boolean(results.error) || (!archived.has(item.key) && !canDelete(item))}
                aria-label={`${archived.has(item.key) ? 'Restore' : 'Delete'} rollout ${item.id}`}
                onClick={event => { event.stopPropagation(); void changeArchive(item, archived.has(item.key)); }}>
                {archived.has(item.key) ? 'Restore' : 'Delete'}</button></td>
            </tr>)}</tbody>
          </table></div>
          {!matchingRuns.length && <p className="run-description">{runs.length ? 'No matching rollouts. Use Show deleted to view removed results.' : 'No rollout results yet. Start a validation from Run & Evaluate.'}</p>}
        </section> : <>
          <dl className="results-summary">
            <div><dt>Requested</dt><dd>{run.targetCount}</dd></div>
            <div><dt>Attempts</dt><dd>{run.attempts.length}</dd></div>
            <div><dt>Released</dt><dd>{run.validation?.released_count ?? run.attempts.filter(item => item.rollout.status?.arm_released).length}</dd></div>
            <div><dt>Judged</dt><dd>{run.validation?.judged_count ?? (run.merged ? (counts?.success ?? 0) + (counts?.failure ?? 0) : 'Not recorded')}</dd></div>
            <div><dt>Mode</dt><dd>{run.mode === 'hold' ? 'Hold · connection test' : run.mode === 'real' ? 'Real inference' : 'Not recorded'}</dd></div>
          </dl>
          <section className="run-panel" aria-label="Policy and execution settings">
            <h2>Policy and execution</h2>
            {run.merged && <p className="run-description">{run.instruction}</p>}
            <dl className="run-details">
              <div><dt>Checkpoint / policy</dt><dd>{run.checkpoint}</dd></div>
              <div><dt>Time limit</dt><dd>{run.merged ? `${run.merged.condition.time_limit_s} s` : run.validation ? `${run.validation.session.configuration.time_limit_s} s` : typeof attempt?.rollout.job.spec.metadata?.time_limit_s === 'number' ? `${attempt.rollout.job.spec.metadata.time_limit_s} s` : 'Not recorded'}</dd></div>
              <div><dt>Maximum relative target</dt><dd>{run.merged?.condition.max_relative_target ?? run.validation?.session.configuration.max_relative_target ?? String(attempt?.rollout.job.spec.metadata?.max_relative_target ?? 'Not recorded')}</dd></div>
              {(run.validation || run.merged) && <div><dt>Outcomes</dt><dd>{counts?.success} success · {counts?.failure} failure · {counts?.unjudged} unjudged</dd></div>}
            </dl>
            {run.mode === 'hold' && <p className="run-note">Hold tests communication with simulated responses; no model inference.</p>}
            {run.validation?.state === 'ended' && <p className="run-note">Validation ended early. The original target and all saved attempts are preserved.</p>}
            {run.validation?.start_errors.map(item => <p key={item.request_id} role="alert">Attempt start error: {item.error}</p>)}
            {Boolean(run.validation?.unresolved_requests.length) && <p role="alert">An attempt start is unconfirmed. No task outcome is inferred.</p>}
            {run.merged && <details className="results-provenance"><summary>Merge sources and provenance</summary><pre className="run-log">{JSON.stringify(run.merged, null, 2)}</pre></details>}
            {run.validation && <details className="results-provenance"><summary>Validation configuration and provenance</summary><pre className="run-log">{JSON.stringify(run.validation.session, null, 2)}</pre></details>}
          </section>
          <div className="results-attempt-browser">
            <section className="run-panel results-attempts" aria-label="Attempt list">
              <div className="run-panel-header"><h2>Attempts</h2><span className="run-status">{run.attempts.length}</span></div>
              <ul className="results-attempt-list">{run.attempts.map(item => <li key={item.rollout.job.id}>
                <button type="button" className={`results-attempt-button ${attempt?.rollout.job.id === item.rollout.job.id ? 'is-active' : ''}`}
                  aria-pressed={attempt?.rollout.job.id === item.rollout.job.id} disabled={blocked}
                  onClick={() => choose({ runKey: run.key, jobId: item.rollout.job.id })}>
                  <span className="results-attempt-number">{item.number}</span><span>Attempt {item.number}<small>{item.rollout.state} · {run.validation || run.merged ? item.outcome : 'Outcome not recorded'}</small></span>
                </button>
              </li>)}</ul>
              {!run.attempts.length && <p className="run-description">No recorded attempts.</p>}
            </section>
            {attempt ? <RolloutResultDetail key={`${run.key}:${attempt.rollout.job.id}`} run={run} attempt={attempt}
              results={results} navigationBlocked={blocked} readOnly={deleted} onDirtyChange={value => setDirtyReview({ id: attempt.rollout.job.id, dirty: value })}
              onOpenExecution={() => { if (!blocked) onOpenExecution({ validationId: run.validation?.id, jobId: attempt.rollout.job.id }); }} />
              : <section className="run-panel"><h2>Attempt details</h2><p>No attempt was started in this rollout.</p></section>}
          </div>
        </>}
      </div>}
  </main>;
}
