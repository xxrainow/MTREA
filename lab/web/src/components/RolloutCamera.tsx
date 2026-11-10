import { useEffect, useState } from 'react';

export default function RolloutCamera({ jobId, name }: { jobId: string; name: string }) {
  const [frame, setFrame] = useState<{ url: string; message: string }>({ url: '', message: 'Waiting for observation…' });
  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController;
    let objectUrl = '';
    async function refresh() {
      controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 2000);
      try {
        const response = await fetch(`/api/rollouts/${encodeURIComponent(jobId)}/frames/${encodeURIComponent(name)}`,
          { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error('Frame unavailable');
        const blob = await response.blob();
        if (disposed) return;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(blob);
        setFrame({ url: objectUrl, message: '' });
      } catch {
        if (!disposed) setFrame({ url: '', message: 'No fresh observation · retrying…' });
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = '';
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
  }, [jobId, name]);
  return <figure className="run-camera">
    <figcaption>{name} · observation frames</figcaption>
    {frame.url ? <img src={frame.url} alt={`${name} rollout camera`} className="run-camera-image"
      onError={() => setFrame({ url: '', message: 'Cannot decode observation frame' })} />
      : <div className="run-camera-placeholder" role="status">{frame.message}</div>}
  </figure>;
}
