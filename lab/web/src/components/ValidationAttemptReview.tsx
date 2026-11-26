import { useState } from 'react';
import type { Outcome, RolloutValidation, ValidationAttempt } from '../api/rolloutValidations';

export default function ValidationAttemptReview({ validation, attempt, validations, blocked, onDirtyChange }: {
  validation: RolloutValidation; attempt: ValidationAttempt;
  validations: {
    busy: boolean;
    annotate: (id: string, jobId: string, outcome: Outcome, notes: string) => Promise<boolean>;
  };
  blocked: boolean;
  onDirtyChange: (dirty: boolean) => void;
}) {
  const [outcome, setOutcome] = useState<Outcome>(attempt.outcome);
  const [notes, setNotes] = useState(attempt.notes);
  const [saved, setSaved] = useState(false);
  const terminal = ['stopped', 'aborted', 'failed', 'ended_unknown'].includes(attempt.rollout.state)
    && !attempt.rollout.read_error;
  const changed = outcome !== attempt.outcome || notes !== attempt.notes;

  async function save() {
    setSaved(false);
    const result = await validations.annotate(validation.id, attempt.rollout.job.id, outcome, notes);
    setSaved(result);
    if (result) onDirtyChange(false);
  }

  return <section className="run-panel rollout-outcome" aria-label="Attempt outcome">
    <h2>Attempt {attempt.number} outcome</h2>
    <p className="run-description">Task success is judged manually. Time limits and execution errors do not assign an outcome.</p>
    <div className="run-number-fields">
      <label className="run-field">Task outcome
        <select value={outcome} disabled={!terminal || blocked} onChange={event => {
          setOutcome(event.target.value as Outcome); setSaved(false);
          onDirtyChange(event.target.value !== attempt.outcome || notes !== attempt.notes);
        }}>
          <option value="unjudged">Unjudged</option><option value="success">Success</option><option value="failure">Failure</option>
        </select>
      </label>
      <label className="run-field">Notes
        <textarea value={notes} rows={2} maxLength={10000} disabled={!terminal || blocked}
          onChange={event => {
            setNotes(event.target.value); setSaved(false);
            onDirtyChange(outcome !== attempt.outcome || event.target.value !== attempt.notes);
          }} />
      </label>
    </div>
    <button type="button" className="run-button" disabled={!terminal || blocked || !changed} onClick={() => void save()}>Save outcome</button>
    {changed && <button type="button" className="run-button" disabled={validations.busy} onClick={() => {
      setOutcome(attempt.outcome); setNotes(attempt.notes); setSaved(false); onDirtyChange(false);
    }}>Discard unsaved edits</button>}
    {saved && !changed && <p className="run-note" role="status">Outcome saved.</p>}
    {!terminal && <p className="run-description">Stop the attempt before recording its outcome.</p>}
  </section>;
}
