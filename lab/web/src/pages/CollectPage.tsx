import { useEffect, useState } from 'react';

import CameraPreviewPanel from '../components/CameraPreviewPanel';
import RecordingView from '../components/RecordingView';
import { useCameras } from '../hooks/useCameras';
import { useCollection } from '../hooks/useCollection';

import './CollectPage.css';

type Props = {
  onBusyChange?: (busy: boolean) => void;
};

function CollectPage({ onBusyChange }: Props) {
  const [datasetName, setDatasetName] = useState('pick_place_demo');
  const [instruction, setInstruction] = useState(
    'Pick up the red block and place it in the blue cup.',
  );
  const [episodeCount, setEpisodeCount] = useState('10');
  const [episodeSeconds, setEpisodeSeconds] = useState('20');
  const [resetSeconds, setResetSeconds] = useState('5');
  const [seed, setSeed] = useState('');

  const [formError, setFormError] = useState('');
  const [showRecording, setShowRecording] = useState(false);

  const collection = useCollection();
  const cameras = useCameras();
  const cameraStatus = cameras.data?.status;

  const busy = collection.active || collection.pending !== null;
  const inputsLocked = busy;
  const isFormValid =
    [
      datasetName,
      instruction,
      episodeCount,
      episodeSeconds,
      resetSeconds,
      seed,
    ].every((value) => value.trim() !== '') &&
    /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(datasetName) &&
    Number.isSafeInteger(Number(episodeCount)) &&
    Number(episodeCount) > 0 &&
    Number.isFinite(Number(episodeSeconds)) &&
    Number(episodeSeconds) > 0 &&
    Number.isFinite(Number(resetSeconds)) &&
    Number(resetSeconds) >= 0 &&
    Number.isSafeInteger(Number(seed));

  const canStart = isFormValid && !busy && !collection.unavailable;

  useEffect(() => {
    onBusyChange?.(busy);
  }, [busy, onBusyChange]);

  useEffect(() => {
    if (!busy) return;

    function warnBeforeLeaving(event: BeforeUnloadEvent) {
      event.preventDefault();
      event.returnValue = '';
    }

    window.addEventListener('beforeunload', warnBeforeLeaving);

    return () => {
      window.removeEventListener('beforeunload', warnBeforeLeaving);
    };
  }, [busy]);

  async function handleStart() {
    if (!canStart) return;

    setFormError('');

    const fields = [
      datasetName,
      instruction,
      episodeCount,
      episodeSeconds,
      resetSeconds,
      seed,
    ];

    if (fields.some((value) => value.trim() === '')) {
      setFormError('Fill in all collection settings.');
      return;
    }

    const count = Number(episodeCount);
    const duration = Number(episodeSeconds);
    const reset = Number(resetSeconds);
    const parsedSeed = Number(seed);

    if (!Number.isSafeInteger(count) || count <= 0) {
      setFormError('Episode count must be a positive integer.');
      return;
    }

    if (
      !Number.isFinite(duration) ||
      duration <= 0 ||
      !Number.isFinite(reset) ||
      reset < 0
    ) {
      setFormError(
        'Recording duration must be positive and reset duration cannot be negative.',
      );
      return;
    }

    if (!Number.isSafeInteger(parsedSeed)) {
      setFormError('Seed must be a safe integer.');
      return;
    }

    const accepted = await collection.start({
      dataset: datasetName,
      seed: parsedSeed,
      session_index: 1,
      sessions: [
        {
          instruction,
          num_episodes: count,
          episode_time_s: duration,
          reset_time_s: reset,
        },
      ],
    });
    if (accepted) setShowRecording(true);
  }

  if (showRecording || busy) {
    return (
      <RecordingView
        status={collection.status}
        receivedAt={collection.receivedAt}
        active={collection.active}
        pending={collection.pending}
        unavailable={collection.unavailable}
        error={collection.actionError || collection.pollError}
        onStop={() => void collection.stop()}
        onBack={() => setShowRecording(false)}
      />
    );
  }

  return (
    <main className="collect-page">
      <div className="collect-heading">
        <div>
          <h1>Collect</h1>
          <p>Record demonstrations through teleoperation.</p>
        </div>

        <button
          type="button"
          className="collect-button collect-button-primary"
          onClick={handleStart}
          disabled={!canStart}
        >
          {collection.pending === 'start' ? 'Starting…' : 'Start recording'}
        </button>
      </div>

      <div className="collect-layout">
        <div className="collect-main">
          <CameraPreviewPanel cameras={cameras} />

          <section className="collect-panel">
            <div className="collect-panel-header">
              <h2>Recording status</h2>
              <span className="collect-status">
                {collection.pending
                  ? `${collection.pending} requested`
                  : collection.unavailable
                    ? 'Status unavailable'
                    : collection.status?.phase}
              </span>
            </div>

            <dl className="collect-details">
              <div>
                <dt>Server status</dt>
                <dd>
                  {collection.unavailable
                    ? 'Unavailable'
                    : (collection.status?.phase ?? 'Checking…')}
                </dd>
              </div>

              <div>
                <dt>Stop requested</dt>
                <dd>
                  {collection.status
                    ? collection.status.stop_requested
                      ? 'Yes'
                      : 'No'
                    : '—'}
                </dd>
              </div>
            </dl>

            <div className="collect-actions">
              <button
                type="button"
                className="collect-button"
                onClick={() => void collection.stop()}
                disabled={
                  collection.pending !== null ||
                  !collection.active ||
                  collection.status?.phase === 'stopping' ||
                  collection.status?.phase === 'verifying'
                }
              >
                {collection.pending === 'stop' ? 'Stopping…' : 'Stop recording'}
              </button>
            </div>

            <div aria-live="polite">
              {[collection.pollError, collection.actionError]
                .filter(Boolean)
                .map((error, index) => (
                  <p key={index} role="alert">
                    {error}
                  </p>
                ))}

              {collection.status?.stop_requested && (
                <p className="collect-session-guide">
                  A stop was requested. Check the final status and saved
                  results.
                </p>
              )}
            </div>
          </section>

          <section className="collect-panel">
            <div className="collect-panel-header">
              <h2>Recording result</h2>
            </div>

            <pre className="collect-log">
              {collection.unavailable
                ? 'Server status is unavailable.'
                : (collection.status?.error ??
                  (collection.status?.phase === 'finished'
                    ? 'Recording, metadata verification, and history recording have finished.'
                    : 'Waiting for a recording result.'))}
            </pre>
          </section>
        </div>

        <section className="collect-panel">
          <div className="collect-panel-header">
            <h2>Collection settings</h2>
          </div>

          <p className="collect-session-guide">
            Prepare the scene, enter the instruction, and start recording.
            Episodes repeat automatically.
          </p>

          <fieldset className="collect-settings-fields" disabled={inputsLocked}>
            <label className="collect-field">
              Dataset name
              <input
                type="text"
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
              />
            </label>

            <label className="collect-field">
              Seed
              <input
                type="number"
                step="1"
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
              />
            </label>

            <label className="collect-field">
              Task instruction
              <textarea
                rows={4}
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
              />
            </label>

            <label className="collect-field">
              Number of episodes
              <input
                type="number"
                min="1"
                step="1"
                value={episodeCount}
                onChange={(event) => setEpisodeCount(event.target.value)}
              />
            </label>

            <div className="collect-field-pair">
              <label className="collect-field">
                Recording duration (seconds)
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={episodeSeconds}
                  onChange={(event) => setEpisodeSeconds(event.target.value)}
                />
              </label>

              <label className="collect-field">
                Reset duration (seconds)
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={resetSeconds}
                  onChange={(event) => setResetSeconds(event.target.value)}
                />
              </label>
            </div>
          </fieldset>

          {formError && (
            <p className="collect-validation-message" role="alert">
              {formError}
            </p>
          )}

          <dl className="collect-details">
            <div>
              <dt>Recording mode</dt>
              <dd>Configured by the backend</dd>
            </div>

            <div>
              <dt>Cameras</dt>
              <dd>
                {cameras.unavailable
                  ? 'Status unavailable'
                  : cameraStatus?.error
                    ? 'Preview error'
                    : cameraStatus?.running
                      ? cameraStatus.cameras.join(', ')
                      : 'Preview stopped'}
              </dd>
            </div>
          </dl>
        </section>
      </div>
    </main>
  );
}

export default CollectPage;
