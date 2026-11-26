import { useEffect, useState } from 'react';
import { rolloutOwnsDevices } from '../api/rollouts';
import { validationStateLabels } from '../api/rolloutValidations';
import { useRolloutResults } from '../hooks/useRolloutResults';
import { formatElapsed, formatResultDate, listRolloutResults, outcomeCounts, resultTargetSelection } from '../rollouts/results';
import type { ResultSelection, RolloutResultTarget } from '../rollouts/results';
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
  const [dirtyReview, setDirtyReview] = useState({ id: '', dirty: false });
  const runs = results.data ? listRolloutResults(results.data.validations, results.data.trials) : [];
  const run = runs.find(item => item.key === selection.runKey)
    ?? (selection.runKey && selection.jobId ? runs.find(item => item.attempts.some(trial => trial.rollout.job.id === selection.jobId)) : undefined);
  const attempt = run?.attempts.find(item => item.rollout.job.id === selection.jobId) ?? run?.attempts[0];
  const dirty = dirtyReview.id === attempt?.rollout.job.id && dirtyReview.dirty;
  const blocked = dirty || results.saving;
  const counts = run ? outcomeCounts(run.attempts) : null;
  const query = search.toLocaleLowerCase().trim();
  const matchingRuns = runs.filter(item => [item.instruction, item.checkpoint, item.id]
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
    setSelection(next);
  }

  return <main className="run-page rollout-results-page">
    <div className="run-heading results-heading">
      <div><h1>Rollout results</h1><p>Each rollout contains all of its validation attempts. Earlier single rollouts contain one attempt.</p></div>
      <button type="button" className="run-button" disabled={blocked} onClick={results.refresh}>Refresh</button>
    </div>
    {results.error && <p role="alert">Cannot refresh results: {results.error}{results.data ? '. Last loaded records are shown; editing is disabled.' : ''}</p>}
    {dirty && <p className="run-note" role="status">Save or discard outcome edits before switching records or leaving this page.</p>}
    {selection.runKey && results.data && !run && <p role="status">The requested result is unavailable. Select a saved validation below.</p>}
    {results.data === null ? <p>{results.error ? 'Use Refresh to retry loading results.' : 'Loading rollout results…'}</p>
      : runs.length === 0 ? <section className="run-panel"><h2>No rollout results yet</h2><p className="run-description">Start a validation from Run & Evaluate. Saved records will appear here.</p>
        <button type="button" className="run-button" onClick={() => onOpenExecution({})}>Open Run & Evaluate</button></section>
        : <div className="results-content">
          {!run ? <section className="run-panel" aria-label="Rollout runs">
            <div className="run-panel-header"><h2>Rollouts</h2><span className="run-status">{runs.length} rollouts</span></div>
            <label className="run-field results-search">Search rollouts
              <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Instruction, checkpoint or rollout ID" />
            </label>
            <div className="results-table-wrapper"><table className="results-rollout-table">
              <thead><tr><th>Recorded</th><th>Instruction</th><th>Checkpoint / policy</th><th>Attempts</th><th>Status</th><th>Details</th></tr></thead>
              <tbody>{matchingRuns.map(item => <tr key={item.key}>
                <td><span>{formatResultDate(item.createdAt)}</span><small title={item.id}>Rollout {item.id}</small></td>
                <td className="results-instruction-text">{item.instruction || '(Empty saved instruction)'}</td>
                <td>{item.checkpoint}<small>{item.mode === 'hold' ? 'Hold · connection test' : item.mode === 'real' ? 'Real inference' : 'Mode not recorded'}</small></td>
                <td className="results-count">{item.attempts.length} / {item.targetCount}<small>recorded / requested</small></td>
                <td><span className="run-status">{item.validation ? validationStateLabels[item.validation.state] : item.state}</span></td>
                <td><button type="button" className="run-button" disabled={blocked} aria-label={`View rollout ${item.id}`}
                  onClick={() => choose({ runKey: item.key })}>View</button></td>
              </tr>)}</tbody>
            </table></div>
            {!matchingRuns.length && <p className="run-description">No matching rollouts.</p>}
          </section> : <>
            <section className="run-panel" aria-label="Rollout details">
              <div className="run-panel-header"><h2 className="results-instruction-text">{run.instruction}</h2>
                <span className="run-status">{run.attempts.length} / {run.targetCount} attempts</span></div>
                <button type="button" className="run-button" disabled={blocked} onClick={() => choose({})}>← Rollouts</button>
                <div className="run-panel-header results-run-heading"><h3>{formatResultDate(run.createdAt)}</h3><span className="run-status">{run.validation ? validationStateLabels[run.validation.state] : run.state}</span></div>
                <dl className="run-details">
                  <div><dt>Checkpoint / policy</dt><dd>{run.checkpoint}</dd></div>
                  <div><dt>Run ID</dt><dd>{run.id}</dd></div>
                  <div><dt>Mode</dt><dd>{run.mode === 'hold' ? 'Hold · communication testing, no model inference' : run.mode === 'real' ? 'Real inference' : 'Not recorded'}</dd></div>
                  <div><dt>Attempts</dt><dd>{run.validation ? `${run.validation.started_count} / ${run.targetCount} started · ${run.validation.released_count} released · ${run.validation.judged_count} judged` : '1 / 1 recorded · outcome not recorded'}</dd></div>
                  <div><dt>Time limit</dt><dd>{run.validation ? `${run.validation.session.configuration.time_limit_s} s` : typeof attempt?.rollout.job.spec.metadata?.time_limit_s === 'number' ? `${attempt.rollout.job.spec.metadata.time_limit_s} s` : 'Not recorded'}</dd></div>
                  <div><dt>Maximum relative target</dt><dd>{run.validation?.session.configuration.max_relative_target ?? String(attempt?.rollout.job.spec.metadata?.max_relative_target ?? 'Not recorded')}</dd></div>
                  {run.validation && <div><dt>Outcomes</dt><dd>{counts?.success} success · {counts?.failure} failure · {counts?.unjudged} unjudged</dd></div>}
                </dl>
                {run.validation?.state === 'ended' && <p className="run-note">Validation ended early. The original target and all saved attempts are preserved.</p>}
                {run.validation?.start_errors.map(item => <p key={item.request_id} role="alert">Attempt start error: {item.error}</p>)}
                {Boolean(run.validation?.unresolved_requests.length) && <p role="alert">An attempt start is unconfirmed. No task outcome is inferred.</p>}
                <div className="results-table-wrapper"><table className="results-attempt-table">
                  <thead><tr><th>Attempt</th><th>Execution</th><th>Outcome</th><th>Elapsed</th><th>Details</th></tr></thead>
                  <tbody>{run.attempts.map(item => <tr key={item.rollout.job.id} className={item.rollout.job.id === attempt?.rollout.job.id ? 'is-selected' : ''}>
                    <td>#{item.number}</td><td>{item.rollout.state}{rolloutOwnsDevices(item.rollout) ? ' · robot owned' : ''}</td>
                    <td>{run.validation ? item.outcome : 'Not recorded'}</td><td>{formatElapsed(item.rollout.status?.elapsed_s)}</td>
                    <td><button type="button" className="run-button" disabled={blocked} aria-pressed={item.rollout.job.id === attempt?.rollout.job.id}
                      onClick={() => choose({ runKey: run.key, jobId: item.rollout.job.id })}>View attempt {item.number}</button></td>
                  </tr>)}</tbody>
                </table></div>
                {!run.attempts.length && <p className="run-description">No recorded attempt was started in this validation.</p>}
                {run.validation && <details className="results-provenance"><summary>Validation configuration and provenance</summary><pre className="run-log">{JSON.stringify(run.validation.session, null, 2)}</pre></details>}
            </section>
            {run && attempt && <RolloutResultDetail key={`${run.key}:${attempt.rollout.job.id}`} run={run} attempt={attempt}
              results={results} navigationBlocked={blocked} onDirtyChange={value => setDirtyReview({ id: attempt.rollout.job.id, dirty: value })}
              onOpenExecution={() => { if (!blocked) onOpenExecution({ validationId: run.validation?.id, jobId: attempt.rollout.job.id }); }} />}
          </>}
        </div>}
  </main>;
}
