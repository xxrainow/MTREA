import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadRollouts } from '../api/rollouts';
import type { RolloutSnapshot } from '../api/rollouts';
import { loadValidations, saveValidationOutcome } from '../api/rolloutValidations';
import type { Outcome, RolloutValidation } from '../api/rolloutValidations';

type Snapshot = { validations: RolloutValidation[]; trials: RolloutSnapshot[] };

export function useRolloutResults() {
  const [data, setData] = useState<Snapshot | null>(null);
  const [error, setError] = useState('');
  const [saveError, setSaveError] = useState<{ jobId: string; message: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const version = useRef(0);
  const writing = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      const requested = version.current;
      try {
        const [validations, trials] = await Promise.all([loadValidations(controller.signal), loadRollouts(controller.signal)]);
        if (!controller.signal.aborted && requested === version.current && !writing.current) {
          setData({ validations, trials });
          setError('');
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [refreshKey]);

  async function annotate(id: string, jobId: string, outcome: Outcome, notes: string) {
    if (writing.current || error) return false;
    writing.current = true;
    version.current += 1;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await saveValidationOutcome(id, jobId, outcome, notes);
      setData(previous => previous ? { ...previous,
        validations: previous.validations.map(item => item.id === id ? updated : item),
      } : previous);
      return true;
    } catch (reason) {
      setSaveError({ jobId, message: errorMessage(reason) });
      return false;
    } finally {
      writing.current = false;
      version.current += 1;
      setSaving(false);
    }
  }

  return { data, error, saveError, saving, annotate, refreshKey, refresh: () => setRefreshKey(key => key + 1) };
}
