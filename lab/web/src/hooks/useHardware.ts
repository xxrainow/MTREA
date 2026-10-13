import { useCallback, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/hardware';
import type { JointFrame } from '../api/hardware';

export function useHardwarePoll<T>(load: (signal: AbortSignal) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const result = await load(controller.signal);
        if (!controller.signal.aborted) {
          setData(result);
          setError(null);
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(errorMessage(failure));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 1000);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [load, revision]);

  return { data, error, refresh };
}

export function useHardwareAction(refresh: () => void) {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  async function act(label: string, action: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(label);
    setError(null);
    try {
      await action();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      inFlight.current = false;
      setPending(null);
      refresh();
    }
  }
  return { act, pending, error };
}

export function useJointStream() {
  const [frame, setFrame] = useState<JointFrame | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    let expiry: ReturnType<typeof setTimeout>;
    function connect() {
      const url = new URL('/ws/joints', window.location.href);
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(url);
      socket.onopen = () => { if (!disposed) setConnected(true); };
      socket.onmessage = (event) => {
        if (disposed) return;
        try {
          const next: JointFrame = JSON.parse(event.data);
          if (next.type !== 'teleop' && next.type !== 'calibration') return;
          setFrame(next);
          clearTimeout(expiry);
          expiry = setTimeout(() => setFrame(null), 2000);
        } catch {
          setFrame(null);
        }
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => {
        if (disposed) return;
        setConnected(false);
        setFrame(null);
        retry = setTimeout(connect, 2000);
      };
    }
    connect();
    return () => {
      disposed = true;
      clearTimeout(retry);
      clearTimeout(expiry);
      socket.close();
    };
  }, []);

  return { frame, connected };
}
