import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import {
  loadRollouts, loadRolloutCommand, rolloutOwnsDevices, sendRolloutCommand, startRollout,
} from '../api/rollouts';
import type { RolloutRequest, RolloutSnapshot } from '../api/rollouts';

type PendingCommand = { jobId: string; id: string; action: 'stop' | 'release' };
const selectionKey = 'lab.rollout.selected';
const commandKey = 'lab.rollout.command';
function savedCommand(): PendingCommand | null {
  try {
    const item = JSON.parse(sessionStorage.getItem(commandKey) ?? 'null');
    return item && typeof item.jobId === 'string' && typeof item.id === 'string'
      && ['stop', 'release'].includes(item.action) ? item : null;
  } catch { return null; }
}

export function useRollouts() {
  const [items, setItems] = useState<RolloutSnapshot[] | null>(null);
  const [selectedId, setSelectedId] = useState(() => sessionStorage.getItem(selectionKey) ?? '');
  const [pollError, setPollError] = useState('');
  const [actionError, setActionError] = useState('');
  const [posting, setPosting] = useState(false);
  const [pending, setPending] = useState<PendingCommand | null>(savedCommand);
  const [notice, setNotice] = useState('');
  const requestBusy = useRef(false);
  const version = useRef(0);
  const owner = items?.find(rolloutOwnsDevices) ?? null;
  const selected = owner ?? items?.find((item) => item.job.id === selectedId) ?? items?.[0] ?? null;

  function select(id: string) {
    sessionStorage.setItem(selectionKey, id);
    setSelectedId(id);
  }

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      const requestedVersion = version.current;
      try {
        const next = await loadRollouts(controller.signal);
        if (!controller.signal.aborted && !requestBusy.current && requestedVersion === version.current) {
          setItems(next);
          setPollError('');
        }
      } catch (error) {
        if (!controller.signal.aborted) setPollError(errorMessage(error));
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 500);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!pending) return;
    const controller = new AbortController();
    let timer: number | undefined;
    function finish(next: RolloutSnapshot[], state: string, error?: string | null) {
      if (!pending || controller.signal.aborted) return;
      version.current += 1;
      setItems(next);
      setPollError('');
      setActionError(state === 'completed' ? '' : error ?? `${pending.action} ${state}`);
      setNotice(`${pending.action === 'stop' ? 'Stop' : 'Release'} ${state}.`);
      sessionStorage.removeItem(commandKey);
      setPending(null);
    }
    async function poll() {
      if (!pending) return;
      try {
        let result = await loadRolloutCommand(pending.jobId, pending.id, controller.signal);
        const next = await loadRollouts(controller.signal);
        if (controller.signal.aborted) return;
        version.current += 1;
        setItems(next);
        const target = next.find((item) => item.job.id === pending.jobId);
        if (!['completed', 'rejected', 'failed'].includes(result.state)
          && (!target || !rolloutOwnsDevices(target))) {
          // The owner may have exited between reading its acknowledgement and status.
          result = await loadRolloutCommand(pending.jobId, pending.id, controller.signal);
          if (!['completed', 'rejected', 'failed'].includes(result.state)) {
            finish(next, 'unconfirmed', 'The owner ended before acknowledging this command. Inspect the run status.');
            return;
          }
        }
        if (['completed', 'rejected', 'failed'].includes(result.state)) {
          finish(next, result.state, result.error);
          return;
        }
        setNotice(`${pending.action === 'stop' ? 'Stop' : 'Release'} ${result.state}; waiting for completion…`);
        setActionError('');
      } catch (error) {
        if (!controller.signal.aborted) {
          setActionError(`Cannot verify command: ${errorMessage(error)}`);
          try {
            const next = await loadRollouts(controller.signal);
            const target = next.find((item) => item.job.id === pending.jobId);
            if (!target || !rolloutOwnsDevices(target)) {
              finish(next, 'unconfirmed', 'Command acknowledgement is unavailable and the owner has ended. Inspect the run status.');
              return;
            }
          } catch { /* Keep waiting while the backend is unreachable. */ }
        }
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 300);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [pending]);

  async function start(body: RolloutRequest) {
    if (requestBusy.current || pending || owner) return false;
    requestBusy.current = true;
    version.current += 1;
    setPosting(true);
    setActionError('');
    setNotice('');
    try {
      const next = await startRollout(body);
      setItems((previous) => [next, ...(previous ?? []).filter((item) => item.job.id !== next.job.id)]);
      select(next.job.id);
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    } finally {
      version.current += 1;
      requestBusy.current = false;
      setPosting(false);
    }
  }

  async function command(action: 'stop' | 'release') {
    if (!selected || requestBusy.current || pending) return;
    requestBusy.current = true;
    version.current += 1;
    setPosting(true);
    setActionError('');
    try {
      const queued = await sendRolloutCommand(selected.job.id, action);
      const next = { jobId: selected.job.id, id: queued.command_id, action };
      sessionStorage.setItem(commandKey, JSON.stringify(next));
      setPending(next);
      setNotice(`${action === 'stop' ? 'Stop' : 'Release'} queued; waiting for completion…`);
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      version.current += 1;
      requestBusy.current = false;
      setPosting(false);
    }
  }

  return { items, selected, owner, select, start, command, posting, pending, notice,
    pollError, actionError, busy: posting || pending !== null };
}
