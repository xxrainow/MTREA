import { useEffect, useRef, useState } from 'react';

import {
  loadCollectionStatus,
  sendCollectionCommand,
  startCollection,
  stopCollection,
  type CollectionCommandInput,
  type CollectionStartInput,
  type CollectionStatus,
} from '../api/collection';

type Action = 'start' | 'stop' | 'advance' | 'retry';

function message(error: unknown): string {
  return error instanceof Error
    ? error.message
    : 'Collection request failed.';
}

export function useCollection() {
  const [status, setStatus] = useState<CollectionStatus | null>(null);
  const [pollError, setPollError] = useState('');
  const [actionError, setActionError] = useState('');
  const [pending, setPending] = useState<Action | null>(null);
  const [receivedAt, setReceivedAt] = useState(() => performance.now());

  const busy = useRef(false);
  const version = useRef(0);
  const [advancedKey, setAdvancedKey] = useState('');
  const advancedPhase = useRef('');

  const phaseKey = status?.phase_id
    ? `${status.recording_id}:${status.phase_id}`
    : '';

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    async function poll() {
      const requestedVersion = version.current;

      try {
        const next = await loadCollectionStatus(controller.signal);

        if (
          !controller.signal.aborted &&
          !busy.current &&
          requestedVersion === version.current
        ) {
          setStatus(next);
          setReceivedAt(performance.now());
          setPollError('');
        }
      } catch (error) {
        if (
          !controller.signal.aborted &&
          !busy.current &&
          requestedVersion === version.current
        ) {
          setPollError(message(error));
        }
      } finally {
        if (!controller.signal.aborted) {
          timer = window.setTimeout(poll, 500);
        }
      }
    }

    void poll();

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  async function perform(
    action: Action,
    request: () => Promise<CollectionStatus>,
  ): Promise<boolean> {
    if (busy.current) return false;

    busy.current = true;
    version.current += 1;
    setPending(action);
    setActionError('');

    try {
      const next = await request();
      setStatus(next);
      setReceivedAt(performance.now());
      setPollError('');
      return true;
    } catch (error) {
      setActionError(message(error));
      return false;
    } finally {
      version.current += 1;
      busy.current = false;
      setPending(null);
    }
  }

  async function sendEpisodeCommand(
    action: CollectionCommandInput['action'],
    pendingAction: 'advance' | 'retry',
  ): Promise<boolean> {
    if (busy.current || advancedPhase.current === phaseKey) {
      return false;
    }

    const phase = status?.episode_phase;

    if (
      !status ||
      pollError !== '' ||
      status.phase !== 'recording' ||
      status.stop_requested ||
      !status.recording_id ||
      !status.phase_id ||
      status.current_episode === null ||
      (phase !== 'preparing' &&
        phase !== 'recording' &&
        phase !== 'resetting')
    ) {
      return false;
    }

    if (
      (action === 'start_episode' && phase === 'recording') ||
      (action !== 'start_episode' && phase !== 'recording')
    ) {
      return false;
    }

    busy.current = true;
    version.current += 1;
    setPending(pendingAction);
    setActionError('');

    try {
      await sendCollectionCommand({
        action,
        recording_id: status.recording_id,
        expected_phase_id: status.phase_id,
        expected_phase: phase,
        expected_episode: status.current_episode,
      });

      advancedPhase.current = phaseKey;
      setAdvancedKey(phaseKey);
      return true;
    } catch (error) {
      setActionError(message(error));
      return false;
    } finally {
      version.current += 1;
      busy.current = false;
      setPending(null);
    }
  }

  const active = status !== null && [
    'starting',
    'recording',
    'stopping',
    'verifying',
  ].includes(status.phase);

  return {
    status,
    receivedAt,
    active,
    pending,
    pollError,
    actionError,
    unavailable: status === null || pollError !== '',
    start: (input: CollectionStartInput) =>
      perform('start', () => startCollection(input)),
    advance: () => sendEpisodeCommand(
      status?.episode_phase === 'recording'
        ? 'finish_episode'
        : 'start_episode',
      'advance',
    ),
    retry: () => sendEpisodeCommand('retry_episode', 'retry'),
    advancePending: phaseKey !== '' && advancedKey === phaseKey,
    stop: () => perform('stop', stopCollection),
  };
}
