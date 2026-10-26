import { useState } from 'react';
import { hardwarePost, jointLabel, loadDevices } from '../api/hardware';
import type { Arm, CalibrationJointStatus } from '../api/hardware';
import { useHardwareAction, useHardwarePoll } from '../hooks/useHardware';
import './DevicesPage.css';

import CameraPreviewPanel from '../components/CameraPreviewPanel';
import { useCameras } from '../hooks/useCameras';

const rangeStatusLabels: Record<string, string> = {
  missing: 'No measurement',
  invalid: 'Invalid range',
  insufficient: 'Move more',
  minimum_met: 'Minimum check passed',
  fixed: 'Fixed range',
};

function CalibrationRangeBar({
  value,
  status,
}: {
  value: CalibrationJointStatus | undefined;
  status: string;
}) {
  if (status === 'fixed') {
    return <span className="device-description">Fixed range</span>;
  }

  const minimum = value?.min;
  const maximum = value?.max;
  const current = value?.current;

  const hasRange =
    minimum != null && maximum != null && current != null && maximum >= minimum;

  const position = hasRange
    ? maximum === minimum
      ? 50
      : Math.max(
          0,
          Math.min(100, ((current - minimum) / (maximum - minimum)) * 100),
        )
    : null;

  const showMarker =
    position !== null && status !== 'unknown' && status !== 'pending';

  return (
    <div className="calibration-range-bar" data-status={status}>
      <div className="calibration-range-values">
        <span>Min {minimum ?? '—'}</span>
        <span>Max {maximum ?? '—'}</span>
      </div>

      <div
        className="calibration-range-track"
        role="img"
        aria-label={
          showMarker
            ? `Measured range ${minimum} to ${maximum}, current ${current}`
            : 'Range measurement unavailable'
        }
      >
        {showMarker && (
          <span
            className="calibration-range-marker"
            style={{ left: `${position}%` }}
          />
        )}
      </div>

      <div className="calibration-range-caption">
        Current {current ?? '—'} · Span {value?.span ?? '—'}
      </div>
    </div>
  );
}

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
                  <th scope="col">Span</th>
                  <th scope="col">Measured range</th>
                  <th scope="col">Check</th>
                </tr>
              </thead>

              <tbody>
                {config?.joint_order.map((joint) => {
                  const value = calibration.joints[joint];
                  const measuringStarted = calibration.state !== 'homing';

                  const status = unavailable
                    ? 'unknown'
                    : measuringStarted
                      ? (value?.range_status ?? 'missing')
                      : 'pending';

                  const label = unavailable
                    ? 'Unconfirmed'
                    : measuringStarted
                      ? (rangeStatusLabels[status] ?? 'No measurement')
                      : 'Not measured yet';

                  return (
                    <tr key={joint}>
                      <th scope="row">{jointLabel(joint)}</th>
                      <td>{value?.current ?? '—'}</td>
                      <td>{value?.min ?? '—'}</td>
                      <td>{value?.max ?? '—'}</td>
                      <td>{value?.span ?? '—'}</td>
                      <td>
                        <CalibrationRangeBar value={value} status={status} />
                      </td>
                      <td>
                        <span
                          className="calibration-range-status"
                          data-status={status}
                          title={value?.range_message}
                        >
                          {label}
                        </span>

                        {!unavailable &&
                          measuringStarted &&
                          value &&
                          !value.range_ok && (
                            <small className="calibration-range-message">
                              {value.range_message}
                            </small>
                          )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {calibration?.state === 'recording_ranges' && !unavailable && (
          <p className="device-description" role="status">
            {calibration.range_check_passed
              ? 'All joints passed the minimum movement check.'
              : 'Some joints need more movement or valid readings.'}
          </p>
        )}

        <p className="device-description">
          Span is Max − Min in raw encoder steps. Passing the minimum movement
          check does not verify full-range coverage.
        </p>
        <div className="device-actions">
          <button
            type="button"
            disabled={
              busy ||
              unavailable ||
              !calibration?.can_next ||
              advancedStep === step ||
              (calibration?.state === 'recording_ranges' &&
                !calibration.range_check_passed)
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
