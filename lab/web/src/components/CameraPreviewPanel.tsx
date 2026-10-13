import { useState } from 'react';
import type { CameraControls } from '../hooks/useCameras';
import './CameraPreviewPanel.css';

type Props = {
  cameras: CameraControls;
};

export default function CameraPreviewPanel({ cameras }: Props) {
  const {
    data,
    pollError,
    actionError,
    pending,
    running,
    unavailable,
    streamVersion,
  } = cameras;

  const [streamError, setStreamError] = useState('');
  const cameraError = data?.status.error;

  function togglePreview() {
    setStreamError('');
    cameras.togglePreview();
  }

  return (
    <section className="camera-preview-panel">
      <div className="camera-preview-header">
        <h2>Camera preview</h2>
        <span>
          {pending
            ? `${pending}…`
            : unavailable
              ? 'Status unavailable'
              : cameraError
                ? 'Preview failed'
                : running
                  ? 'Preview started'
                  : 'Stopped'}
        </span>
      </div>

      <button
        type="button"
        className="camera-preview-button"
        disabled={
          Boolean(pending) || unavailable || (data?.configs.length ?? 0) === 0
        }
        onClick={togglePreview}
      >
        {running ? 'Stop preview' : 'Start preview'}
      </button>

      <div aria-live="polite">
        {pollError && (
          <p className="camera-preview-error" role="alert">
            {pollError}. Retrying…
          </p>
        )}
        {actionError && (
          <p className="camera-preview-error" role="alert">
            {actionError}
          </p>
        )}
        {streamError && !cameraError && (
          <p className="camera-preview-error" role="alert">
            {streamError}
          </p>
        )}
        {cameraError && (
          <p className="camera-preview-error" role="alert">
            {cameraError}. Reconnect the camera and start preview again.
          </p>
        )}
      </div>

      <div className="camera-preview-grid">
        {data?.configs.map((camera) => (
          <figure className="camera-preview-camera" key={camera.name}>
            <figcaption>
              {camera.name} · {camera.width} × {camera.height}
              {' · '}
              {camera.fps} FPS configured
            </figcaption>

            {!pollError && data.status.cameras.includes(camera.name) ? (
              <img
                className="camera-preview-image"
                src={`/api/cameras/${encodeURIComponent(camera.name)}/stream?v=${streamVersion}`}
                alt={`${camera.name} camera preview`}
                onError={() =>
                  setStreamError(
                    `${camera.name}: stream unavailable. Stop and restart preview.`,
                  )
                }
              />
            ) : (
              <div className="camera-preview-placeholder">
                {pollError ? 'Status unavailable' : 'Preview stopped'}
              </div>
            )}
          </figure>
        ))}
      </div>

      <p className="camera-preview-note">
        Preview continues when you change pages. Stop preview here or in Collect
        when finished.
      </p>
    </section>
  );
}
