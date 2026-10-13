import { useState } from 'react';
import { hardwarePost, jointLabel, loadDevices } from '../api/hardware';
import type { Arm } from '../api/hardware';
import { useHardwareAction, useHardwarePoll } from '../hooks/useHardware';
import './DevicesPage.css';

import CameraPreviewPanel from '../components/CameraPreviewPanel';
import { useCameras } from '../hooks/useCameras';

function DevicesPage() {
  const { data, error: pollError, refresh } = useHardwarePoll(loadDevices);
  const { act, pending, error: actionError } = useHardwareAction(refresh);
  const [ports, setPorts] = useState<Partial<Record<Arm, string>>>({});
  const [detection, setDetection] = useState<{
    arm: Arm;
    port: string | null;
  } | null>(null);
  const [advancedStep, setAdvancedStep] = useState<string | null>(null);
  const config = data?.config;
  const calibration = data?.calibration;

  const busy = Boolean(pending);
  const unavailable = !data || Boolean(pollError);
  const held = Boolean(config?.lock);

  const step = calibration ? `${calibration.arm}:${calibration.state}` : '';

  const cameras = useCameras();


  function detect(arm: Arm) {
    void act('Starting port detection', async () => {
      await hardwarePost('/api/hardware/ports/detect/start', { arm });
      setDetection({ arm, port: null });
    });
  }

  function finishDetection() {
    if (!detection) return;
    const arm = detection.arm;
    void act('Waiting for the unplugged port', async () => {
      const result = await hardwarePost<{ port: string }>(
        '/api/hardware/ports/detect/finish',
        { arm },
      );
      setPorts((current) => ({ ...current, [arm]: result.port }));
      setDetection({ arm, port: result.port });
    });
  }

  function savePort(arm: Arm, port: string) {
    void act('Saving port', async () => {
      await hardwarePost('/api/hardware/ports/save', {
        arm,
        port: port.trim(),
      });
      setDetection(null);
    });
  }

  return (
    <main className="devices-page">
      <div className="devices-heading">
        <h1>Devices</h1>
        <p>
          Set arm ports, then calibrate the leader and follower before
          teleoperation.
        </p>
      </div>
      <div className="hardware-feedback" aria-live="polite">
        <p>
          {config
            ? config.backend === 'fake'
              ? 'Fake hardware · simulated arms and separate calibration files'
              : 'Real hardware'
            : 'Loading hardware settings…'}
        </p>
        {pollError && (
          <p role="alert">Status unavailable: {pollError}. Retrying…</p>
        )}
        {actionError && <p role="alert">{actionError}</p>}
        {pending && <p>{pending}…</p>}
        {config?.lock && <p>Arms in use: {config.lock.owner}</p>}
      </div>

      <div className="devices-grid">
        {(['follower', 'leader'] as const).map((arm) => {
          const cfg = config?.arms[arm];
          const port = ports[arm] ?? cfg?.port ?? '';
          return (
            <section className="device-panel" key={arm}>
              <div className="device-panel-header">
                <h2>{arm === 'follower' ? 'Follower' : 'Leader'}</h2>
                <span className="device-status">
                  {unavailable
                    ? 'Unconfirmed'
                    : cfg?.port_present === true
                      ? 'Port present'
                      : cfg?.port_present === false
                        ? 'Port missing'
                        : 'Port not checked'}
                </span>
              </div>
              {cfg?.error && <p role="alert">{cfg.error}</p>}
              <dl className="device-details">
                <div>
                  <dt>Robot type</dt>
                  <dd>SO-101</dd>
                </div>
                <div>
                  <dt>Device ID</dt>
                  <dd>{cfg?.id ?? '—'}</dd>
                </div>
                <div>
                  <dt>Saved port</dt>
                  <dd>{cfg?.port ?? '—'}</dd>
                </div>
                <div>
                  <dt>Calibration</dt>
                  <dd>
                    {unavailable
                      ? 'Unconfirmed'
                      : cfg?.calibration_present
                        ? 'File present'
                        : 'Required'}
                  </dd>
                </div>
                <div>
                  <dt>Joint units</dt>
                  <dd>
                    {cfg?.use_degrees === undefined
                      ? '—'
                      : cfg.use_degrees
                        ? 'Degrees; gripper 0–100'
                        : 'Normalized'}
                  </dd>
                </div>
              </dl>
              {cfg?.legacy_calibration_present && !cfg.calibration_present && (
                <p className="device-description">
                  An older calibration file exists, but this LeRobot version
                  does not read it. Calibrate again.
                </p>
              )}
              <label className="device-port-field">
                Serial port
                <input
                  aria-label={`${arm} serial port`}
                  list="hardware-ports"
                  value={port}
                  disabled={
                    busy ||
                    held ||
                    unavailable ||
                    Boolean(detection && detection.arm !== arm)
                  }
                  onChange={(event) =>
                    setPorts((current) => ({
                      ...current,
                      [arm]: event.target.value,
                    }))
                  }
                />
              </label>
              <div className="device-actions">
                <button
                  type="button"
                  disabled={busy || held || unavailable || Boolean(detection)}
                  onClick={() => detect(arm)}
                >
                  Detect port
                </button>
                <button
                  type="button"
                  disabled={
                    busy ||
                    held ||
                    unavailable ||
                    !port.trim() ||
                    Boolean(
                      detection && (detection.arm !== arm || !detection.port),
                    )
                  }
                  onClick={() => savePort(arm, port)}
                >
                  Save port
                </button>
                <button
                  type="button"
                  disabled={
                    busy ||
                    held ||
                    unavailable ||
                    Boolean(detection) ||
                    Boolean(cfg?.error) ||
                    cfg?.port_present !== true
                  }
                  onClick={() =>
                    void act(`Starting ${arm} calibration`, async () => {
                      await hardwarePost('/api/calibration/start', { arm });
                      setAdvancedStep(null);
                    })
                  }
                >
                  Calibrate
                </button>
              </div>
            </section>
          );
        })}
      </div>
      <datalist id="hardware-ports">
        {config?.ports?.map((port) => (
          <option key={port} value={port} />
        ))}
      </datalist>

      {detection && (
        <section
          className="device-panel devices-calibration"
          aria-live="polite"
        >
          <h2>Detect {detection.arm} port</h2>
          {detection.port ? (
            <p>
              Detected {detection.port}. Plug the {detection.arm} USB cable back
              in, then press Save port above.
            </p>
          ) : (
            <>
              <p>
                Unplug only the {detection.arm} USB cable, then check the
                removed port. Detection waits up to 15 seconds. In fake mode,
                unplugging is simulated.
              </p>
              <button
                type="button"
                disabled={busy || held || unavailable}
                onClick={finishDetection}
              >
                Check removed port
              </button>
            </>
          )}
        </section>
      )}

      <section className="device-panel devices-calibration">
        <div className="device-panel-header">
          <h2>Calibration{calibration?.arm ? ` · ${calibration.arm}` : ''}</h2>
          <span className="device-status">
            {unavailable
              ? 'Unconfirmed'
              : calibration?.state.replaceAll('_', ' ')}
          </span>
        </div>
        <p aria-live="polite">
          {calibration?.instruction ?? 'Loading calibration status…'}
        </p>
        <p className="device-description">
          Joint readings below are raw encoder steps.
        </p>
        {calibration?.active && (
          <p className="device-description">
            Torque is off. Support the arm while moving it by hand.
          </p>
        )}
        {calibration?.error && <p role="alert">{calibration.error}</p>}
        {calibration && Object.keys(calibration.joints).length > 0 && (
          <div className="hardware-table-scroll">
            <table className="hardware-joints">
              <thead>
                <tr>
                  <th scope="col">Joint</th>
                  <th scope="col">Current</th>
                  <th scope="col">Min</th>
                  <th scope="col">Max</th>
                </tr>
              </thead>
              <tbody>
                {config?.joint_order.map((joint) => {
                  const value = calibration.joints[joint];
                  return (
                    <tr key={joint}>
                      <th scope="row">{jointLabel(joint)}</th>
                      <td>{value?.current ?? '—'}</td>
                      <td>{value?.min ?? '—'}</td>
                      <td>{value?.max ?? '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="device-actions">
          <button
            type="button"
            disabled={
              busy ||
              unavailable ||
              !calibration?.can_next ||
              advancedStep === step
            }
            onClick={() =>
              void act('Advancing calibration', async () => {
                await hardwarePost('/api/calibration/next');
                setAdvancedStep(step);
              })
            }
          >
            {calibration?.state === 'recording_ranges'
              ? 'Finish and save'
              : 'Next'}
          </button>
          <button
            type="button"
            disabled={busy || !calibration?.active}
            onClick={() =>
              void act('Cancelling calibration', async () => {
                await hardwarePost('/api/calibration/cancel');
              })
            }
          >
            Cancel
          </button>
        </div>
        <p className="hardware-session-note device-description">
          Calibration continues when you change pages. Return here to finish or
          cancel it.
        </p>
      </section>

      <CameraPreviewPanel cameras={cameras} />
    </main>
  );
}

export default DevicesPage;
