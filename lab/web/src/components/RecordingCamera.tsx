import { useEffect, useState } from 'react';

export default function RecordingCamera({ name, recordingId }: { name: string; recordingId: string }) {
  const [url, setUrl] = useState('');
  const [state, setState] = useState('Waiting for recording frames…');

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController;
    let objectUrl = '';

    async function refresh() {
      controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 2000);
      try {
        const response = await fetch(
          `/api/collection/frames/${encodeURIComponent(recordingId)}/${encodeURIComponent(name)}`,
          { signal: controller.signal, cache: 'no-store' },
        );
        if (!response.ok) throw new Error('Frame unavailable');
        const blob = await response.blob();
        if (disposed) return;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        if (!disposed) setState('No fresh frame · reconnecting…');
      } finally {
        window.clearTimeout(timeout);
        if (!disposed) timer = window.setTimeout(refresh, 300);
      }
    }

    void refresh();
    return () => {
      disposed = true;
      controller?.abort();
      window.clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [name, recordingId]);

  return (
    <figure className="recording-camera recording-live-camera">
      {url && <img
        src={url} alt={`${name} recording camera`}
        onLoad={() => setState('LIVE · RECORDER FRAMES')}
        onError={() => setState('Cannot decode recording frame')}
        style={{ visibility: state.startsWith('LIVE') ? 'visible' : 'hidden' }}
      />}
      {!url && <div className="recording-live-empty" />}
      {!state.startsWith('LIVE') && <div className="recording-live-message" role="status">{state}</div>}
      <figcaption><span>{name}</span><span>{state}</span></figcaption>
    </figure>
  );
}
