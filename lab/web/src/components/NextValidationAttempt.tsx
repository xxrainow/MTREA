import { useState } from 'react';
import type { RolloutValidation } from '../api/rolloutValidations';

export default function NextValidationAttempt({ validation, enabled, onStart }: {
  validation: RolloutValidation; enabled: boolean; onStart: () => void;
}) {
  const [ready, setReady] = useState(false);
  return <div className="rollout-next-attempt">
    <label><input type="checkbox" checked={ready} disabled={!enabled} onChange={event => setReady(event.target.checked)} />
      Scene reset and robot ready</label>
    <button type="button" className="run-button run-button-primary" disabled={!enabled || !ready} onClick={onStart}>
      Start next attempt ({validation.started_count + 1} / {validation.session.target_count})
    </button>
  </div>;
}
