import { useEffect, useRef, useState } from 'react';
import type { EpisodeVideo } from '../api/datasets';

function time(value: number) {
  const seconds = Math.max(0, Math.floor(value));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

export default function DatasetVideoPlayer({ clip, url }: { clip: EpisodeVideo; url: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!playing) return;
    let frame: number;
    function tick() {
      const video = videoRef.current;
      if (!video) return;
      const end = clip.start_s + duration;
      // A v3 MP4 may hold other episodes after this segment.
      if (video.currentTime >= end) {
        video.pause();
        video.currentTime = Math.max(clip.start_s, end - 0.001);
      }
      const offset = video.currentTime - clip.start_s;
      setPosition(offset >= duration - 0.002 ? duration : Math.max(0, offset));
      if (!video.paused) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, clip.start_s, duration]);

  function loaded() {
    const video = videoRef.current;
    if (!video) return;
    const end = clip.end_s ?? video.duration;
    if (!Number.isFinite(end) || end <= clip.start_s || end > video.duration + 0.1) {
      setError('Episode timestamps do not match the video duration.');
      return;
    }
    setDuration(Math.min(end, video.duration) - clip.start_s);
    video.currentTime = clip.start_s;
    setReady(true);
  }

  async function toggle() {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) { video.pause(); return; }
    if (video.currentTime >= clip.start_s + duration - 0.03) video.currentTime = clip.start_s;
    try { await video.play(); }
    catch { setError('Playback failed. Your browser may not support this video codec.'); }
  }

  return (
    <figure className="datasets-video-card">
      <figcaption>{clip.camera}</figcaption>
      <video
        ref={videoRef} preload="metadata" playsInline muted
        src={`${url}#t=${clip.start_s}${clip.end_s === null ? '' : `,${clip.end_s}`}`}
        aria-label={`${clip.camera} episode video`}
        onLoadedMetadata={loaded}
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
        onTimeUpdate={() => {
          const video = videoRef.current;
          if (!video || !ready) return;
          if (video.currentTime >= clip.start_s + duration) {
            video.pause();
            video.currentTime = Math.max(clip.start_s, clip.start_s + duration - 0.001);
          }
          const offset = video.currentTime - clip.start_s;
          setPosition(offset >= duration - 0.002 ? duration : Math.max(0, offset));
        }}
        onError={() => { setReady(false); setError('Cannot load video. The file may be missing or its codec unsupported by this browser.'); }}
      />
      <div className="datasets-player-controls">
        <button type="button" className="datasets-button" disabled={!ready || !!error} onClick={() => void toggle()} aria-label={`${playing ? 'Pause' : 'Play'} ${clip.camera}`}>{playing ? 'Pause' : 'Play'}</button>
        <input
          className="datasets-video-seek" aria-label={`${clip.camera} playback position`} type="range"
          min={0} max={duration || 1} step={0.01} value={position} disabled={!ready || !!error}
          onChange={(event) => {
            const next = Number(event.target.value);
            if (videoRef.current) videoRef.current.currentTime = clip.start_s + Math.min(next, Math.max(0, duration - 0.001));
            setPosition(next);
          }}
        />
        <span className="datasets-player-time">{time(position)} / {time(duration)}</span>
        <select aria-label={`${clip.camera} playback speed`} defaultValue="1" disabled={!ready} onChange={(event) => { if (videoRef.current) videoRef.current.playbackRate = Number(event.target.value); }}>
          <option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option>
        </select>
      </div>
      {!ready && !error && <p className="datasets-note" role="status">Loading video…</p>}
      {error && <p className="datasets-error" role="alert">{error}</p>}
    </figure>
  );
}
