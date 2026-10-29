import { useEffect, useState } from 'react';
import type { CollectionStatus } from '../api/collection';
import './RecordingView.css';
import RecordingCamera from './RecordingCamera';

type Props = {
  status: CollectionStatus | null;
  receivedAt: number;
  active: boolean;
  pending: 'start' | 'stop' | 'advance' | null;
  advancePending: boolean;
  onAdvance: () => void;
  unavailable: boolean;
  error: string;
  onStop: () => void;
  onBack: () => void;
};

function clock(seconds: number) {
  const value = Math.max(0, Math.ceil(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
}

function SimulatedCamera({ name, index }: { name: string; index: number }) {
  return (
    <figure className="recording-camera">
      <svg viewBox="0 0 600 360" role="img" aria-label={`${name}: simulated robot illustration, not live video`}>
        <rect width="600" height="360" fill="#1b303a" />
        <path d="M0 50H600M0 110H600M0 170H600M0 230H600M0 290H600M80 0V360M180 0V360M280 0V360M380 0V360M480 0V360M580 0V360" stroke="#ffffff" opacity=".045" />
        <ellipse cx="310" cy="280" rx="235" ry="48" fill="#0d2028" />
        <rect x={index % 2 ? 280 : 340} y="210" width="47" height="47" rx="5" fill="#f77369" />
        <rect x="420" y="175" width="38" height="38" rx="5" fill="#6587ed" />
        <ellipse cx="490" cy="260" rx="46" ry="19" fill="#53b8a3" />
        <ellipse cx="490" cy="254" rx="46" ry="19" fill="#8dd5c4" />
        <rect x="78" y="275" width="100" height="17" rx="7" fill="#718794" />
        <g className="recording-sim-arm">
          <path d="M130 275L170 175L265 120L340 170" fill="none" stroke="#b19ae3" strokeWidth="24" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="170" cy="175" r="18" fill="#7e65b5" />
          <circle cx="265" cy="120" r="18" fill="#7e65b5" />
          <path d="M340 170L327 198M340 170L361 190" fill="none" stroke="#d7c7f5" strokeWidth="10" strokeLinecap="round" />
        </g>
      </svg>
      <figcaption><span>{name}</span><span>SIMULATED · NO LIVE VIDEO</span></figcaption>
    </figure>
  );
}

export default function RecordingView({
  status,
  receivedAt,
  active,
  pending,
  unavailable,
  error,
  onStop,
  onBack,
  advancePending,
  onAdvance,
}: Props) {
  const [now, setNow] = useState(() => performance.now());

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [active]);

  const timed =
    active &&
    !unavailable &&
    pending !== 'start' &&
    pending !== 'stop' &&
    status?.phase === 'recording';
  const duration = timed ? status?.phase_duration_s ?? null : null;
  const extra = timed ? Math.max(0, now - receivedAt) / 1000 : 0;
  const elapsed = Math.min(duration ?? 0, (status?.phase_elapsed_s ?? 0) + extra);
  const remaining = duration === null ? null : Math.max(0, duration - elapsed);
  const progress = duration && duration > 0 ? Math.min(100, elapsed / duration * 100) : 0;
  const busy = active || pending !== null;
  const stage = status?.episode_phase ?? 'connecting';
  const finished = status?.phase === 'finished';
  const failed = status?.phase === 'failed';

  let title = 'CONNECTING…';
  if (pending === 'start') title = 'CONNECTING…';
  else if (unavailable) title = 'CONNECTION LOST — STATUS UNKNOWN';
  else if (pending === 'stop' || status?.phase === 'stopping') title = 'STOPPING — PLEASE WAIT';
  else if (status?.phase === 'verifying') title = 'VERIFYING SAVED DATA';
  else if (failed) title = status.stop_requested ? 'STOPPED — CHECK SAVED RESULTS' : 'RECORDING FAILED';
  else if (finished) title = status.stop_requested ? 'RECORDING STOPPED' : 'COLLECTION COMPLETE';
  else if (status?.phase === 'recording') {
    title = stage === 'recording' ? `RECORDING EPISODE ${status.current_episode ?? '—'}`
      : stage === 'resetting' ? 'RESET — GET READY'
      : stage === 'preparing' ? 'GET READY'
      : stage === 'saving' ? 'SAVING EPISODE'
      : stage === 'connecting' ? 'CONNECTING TO HARDWARE'
      : 'RECORDING PROCESS ACTIVE';
  }

  const canAdvance =
    active &&
    !unavailable &&
    status?.phase === 'recording' &&
    !status.stop_requested &&
    (stage === 'preparing' ||
      stage === 'recording' ||
      stage === 'resetting');

  const advanceLabel = stage === 'recording'
    ? 'End episode'
    : stage === 'resetting'
      ? 'Start next episode'
      : 'Start now';

  const tone = failed ? 'error' : finished ? 'complete' : stage;

  return (
    <main className="recording-view" data-stage={tone}>
      <div className="recording-toolbar">
        <button type="button" className="collect-button" onClick={onBack} disabled={busy}>← Back to settings</button>
        <span className="recording-mode">{status ? `${status.backend.toUpperCase()} MODE` : 'CONNECTING'}</span>
      </div>

      <section className="recording-stage" aria-label="Recording workspace">
        <header className="recording-stage-header">
          <div><span className="recording-eyebrow">DATA COLLECTION</span><h1>{status?.dataset ?? 'Preparing recording'}</h1></div>
          <div className="recording-counters"><span>Episode <strong>{status?.current_episode ?? '—'}</strong> / {status?.total_episodes ?? '—'}</span><span>{clock((status?.session_elapsed_s ?? 0) + (active && !unavailable ? extra : 0))} elapsed</span></div>
        </header>

        <div className="recording-phase" role="status"><span />{title}</div>
        <p className="recording-instruction">{status?.instruction ?? 'Waiting for the recorder…'}</p>

        <div className="recording-camera-grid">
          {status?.backend === 'fake' ? status.camera_names.map((name, index) => <SimulatedCamera key={name} name={name} index={index} />) : active && status?.recording_id && !unavailable && pending !== 'stop' && status.phase === 'recording' ? status.camera_names.map((name) => (
            <RecordingCamera key={`${status.recording_id}/${name}`} name={name} recordingId={status.recording_id!} />
          )) : (
            <div className="recording-camera-unavailable">{active ? 'Waiting for the recorder connection…' : 'Recording cameras released. View saved videos in Datasets.'}</div>
          )}
        </div>

        <div className="recording-timer" aria-label="Time remaining">
          <span>{remaining !== null ? clock(remaining) : finished ? 'DONE' : failed ? 'STOP' : '—'}</span>
          <small>{duration !== null ? `/ ${clock(duration)} · remaining` : busy ? 'Waiting for recorder progress' : 'Recording ended'}</small>
        </div>

        <div className="recording-progress" role="progressbar" aria-label="Current phase progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={finished ? 100 : Math.round(progress)}><div style={{ width: `${finished ? 100 : progress}%` }} /></div>

        <div className="recording-summary"><span><strong>{status?.saved_episodes ?? 0}</strong> / {status?.total_episodes ?? '—'} episodes saved</span><span>{status?.backend === 'fake' ? 'Simulated frames · metadata only' : 'Recorder progress · shared camera frames'}</span></div>

        {status?.preview_error && <p className="recording-error" role="alert">Preview: {status.preview_error}</p>}

        {(error || status?.error) && <p className="recording-error" role="alert">{error || status?.error}</p>}
        {canAdvance && (
          <button
            type="button"
            className="recording-advance"
            onClick={onAdvance}
            disabled={pending !== null || advancePending}
          >
            {pending === 'advance' || advancePending
              ? 'Waiting for recorder…'
              : advanceLabel}
          </button>
        )}
        {busy ? (
          <button type="button" className="recording-stop" onClick={onStop} disabled={pending !== null || status?.phase === 'stopping' || status?.phase === 'verifying'}>{pending === 'stop' || status?.phase === 'stopping' ? 'Stopping and saving…' : 'Stop recording'}</button>
        ) : <button type="button" className="recording-return" onClick={onBack}>Back to collection settings</button>}
        <p className="recording-hint">{active ? 'Episodes advance automatically. Use the button to end the current phase early.' : 'Saved episodes and collection history are preserved.'}</p>
      </section>
    </main>
  );
}
