import { hardwarePost, jointLabel, loadTeleop } from '../api/hardware';
import {
  useHardwareAction,
  useHardwarePoll,
  useJointStream,
} from '../hooks/useHardware';
import './TeleoperatePage.css';

function TeleoperatePage() {
  const { data, error: pollError, refresh } = useHardwarePoll(loadTeleop);
  const { act, pending, error: actionError } = useHardwareAction(refresh);
  const { frame, connected } = useJointStream();
  const config = data?.config;
  const teleop = data?.teleop;
  const unavailable = !data || Boolean(pollError);
  const ready =
    config &&
    (config.backend === 'fake' ||
      Object.values(config.arms).every(
        (arm) =>
          !arm.error && arm.port_present === true && arm.calibration_present,
      ));
  const joints =
    !unavailable && teleop?.running && connected && frame?.type === 'teleop'
      ? frame
      : null;

  return (
    <main className="teleoperate-page">
      <div className="teleoperate-heading">
        <h1>Teleoperate</h1>
        <p>Move the leader to control the follower.</p>
      </div>
      <div className="hardware-feedback" aria-live="polite">
        <p>
          {config
            ? config.backend === 'fake'
              ? 'Fake hardware · simulated movement'
              : 'Real hardware'
            : 'Loading hardware status…'}
        </p>
        {pollError && (
          <p role="alert">Status unavailable: {pollError}. Retrying…</p>
        )}
        {actionError && <p role="alert">{actionError}</p>}
        {teleop?.error && <p role="alert">{teleop.error}</p>}
        {pending && <p>{pending}…</p>}
      </div>
      <div className="teleoperate-layout">
        <div className="teleoperate-controls">
          <section className="teleoperate-panel">
            <div className="teleoperate-panel-header">
              <h2>Teleoperation control</h2>
              <span className="teleoperate-status">
                {unavailable
                  ? 'Unconfirmed'
                  : teleop?.running
                    ? 'Running'
                    : 'Stopped'}
              </span>
            </div>
            <dl className="teleoperate-details">
              {(['leader', 'follower'] as const).map((arm) => (
                <div key={arm}>
                  <dt>{arm === 'leader' ? 'Leader' : 'Follower'}</dt>
                  <dd>
                    {config?.arms[arm].port ?? '—'} ·{' '}
                    {unavailable
                      ? 'Unconfirmed'
                      : config?.arms[arm].port_present
                        ? 'Port present'
                        : 'Port not confirmed'}
                  </dd>
                </div>
              ))}
              <div>
                <dt>Calibration</dt>
                <dd>
                  {unavailable
                    ? 'Unconfirmed'
                    : Object.values(config!.arms).every(
                          (arm) => arm.calibration_present,
                        )
                      ? 'Both files present'
                      : 'Complete both arms in Devices'}
                </dd>
              </div>
              <div>
                <dt>Control rate</dt>
                <dd>
                  {unavailable
                    ? '—'
                    : `${teleop?.fps} / ${teleop?.target_fps} FPS`}
                </dd>
              </div>
              <div>
                <dt>Arms in use</dt>
                <dd>
                  {unavailable
                    ? 'Unconfirmed'
                    : (config?.lock?.owner ?? 'None')}
                </dd>
              </div>
            </dl>
            <div className="teleoperate-actions">
              <button
                type="button"
                className="teleoperate-start"
                disabled={
                  Boolean(pending) ||
                  unavailable ||
                  !ready ||
                  Boolean(config?.lock) ||
                  Boolean(teleop?.running)
                }
                onClick={() =>
                  void act('Starting teleoperation', async () => {
                    await hardwarePost('/api/teleop/start');
                  })
                }
              >
                Start
              </button>
              <button
                type="button"
                disabled={Boolean(pending) || !teleop?.running}
                onClick={() =>
                  void act('Stopping teleoperation', async () => {
                    await hardwarePost('/api/teleop/stop');
                  })
                }
              >
                Stop
              </button>
            </div>
            <p className="teleoperate-note">
              Before stopping, move the follower to a supported rest pose. Stop
              turns its torque off and the arm goes limp.
            </p>
            <p className="teleoperate-note">
              Teleoperation continues when you change pages. Return here to stop
              it before collecting data.
            </p>
          </section>
          <section className="teleoperate-panel">
            <div className="teleoperate-panel-header">
              <h2>Joint positions</h2>
              <span className="teleoperate-status">
                {joints
                  ? 'Live'
                  : connected
                    ? 'Waiting for frames'
                    : 'Stream reconnecting…'}
              </span>
            </div>
            <div className="hardware-table-scroll">
              <table className="teleoperate-joints">
                <thead>
                  <tr>
                    <th scope="col">Joint</th>
                    <th scope="col">Leader</th>
                    <th scope="col">Follower</th>
                  </tr>
                </thead>
                <tbody>
                  {config?.joint_order.map((joint) => (
                    <tr key={joint}>
                      <th scope="row">{jointLabel(joint)}</th>
                      <td>{joints?.leader[joint]?.toFixed(2) ?? '—'}</td>
                      <td>{joints?.follower[joint]?.toFixed(2) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="teleoperate-note">
              Joint units follow each arm’s use_degrees setting; the gripper
              uses 0–100.
            </p>
          </section>
        </div>
      </div>
    </main>
  );
}

export default TeleoperatePage;
