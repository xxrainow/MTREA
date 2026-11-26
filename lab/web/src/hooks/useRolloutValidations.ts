import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { endValidation, loadValidations, nextValidationAttempt, saveValidationOutcome, startValidation, token } from '../api/rolloutValidations';
import type { Outcome, RolloutValidation, ValidationRequest } from '../api/rolloutValidations';
import type { useRollouts } from './useRollouts';

const selectionKey = 'lab.rollout.validation';

export function useRolloutValidations(rollout: ReturnType<typeof useRollouts>) {
  const [items, setItems] = useState<RolloutValidation[] | null>(null);
  const [selectedId, setSelectedId] = useState(() => sessionStorage.getItem(selectionKey) ?? '');
  const [busy, setBusy] = useState(false);
  const [pollError, setPollError] = useState('');
  const [actionError, setActionError] = useState('');
  const locked = useRef(false);
  const version = useRef(0);
  const active = items?.find(item => !['completed', 'ended'].includes(item.state)) ?? null;
  const selected = active ?? items?.find(item => item.id === selectedId) ?? items?.[0] ?? null;

  function select(id: string) {
    sessionStorage.setItem(selectionKey, id);
    setSelectedId(id);
    const last = items?.find(item => item.id === id)?.attempts.at(-1);
    if (last) rollout.adopt(last.rollout);
  }

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      const requested = version.current;
      try {
        const next = await loadValidations(controller.signal);
        if (!controller.signal.aborted && !locked.current && requested === version.current) {
          setItems(next);
          setPollError('');
        }
      } catch (error) {
        if (!controller.signal.aborted) setPollError(errorMessage(error));
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 1000);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, []);

  async function mutate(action: () => Promise<RolloutValidation>, adopt = false) {
    if (locked.current || rollout.busy) return false;
    locked.current = true;
    version.current += 1;
    setBusy(true);
    setActionError('');
    try {
      const next = await action();
      setItems(previous => [next, ...(previous ?? []).filter(item => item.id !== next.id)]);
      sessionStorage.setItem(selectionKey, next.id);
      setSelectedId(next.id);
      const last = next.attempts.at(-1);
      if (adopt && last) rollout.adopt(last.rollout);
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    } finally {
      version.current += 1;
      locked.current = false;
      setBusy(false);
    }
  }

  return { items, active, selected, busy, pollError, actionError, select,
    start: (body: ValidationRequest) => mutate(() => startValidation(body, token()), true),
    next: (item: RolloutValidation) => mutate(() => nextValidationAttempt(item.id, item.started_count + 1, token()), true),
    end: (item: RolloutValidation) => mutate(() => endValidation(item.id)),
    annotate: (id: string, jobId: string, outcome: Outcome, notes: string) =>
      mutate(() => saveValidationOutcome(id, jobId, outcome, notes)),
  };
}
