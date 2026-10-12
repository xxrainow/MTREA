import './TeleoperatePage.css';

const jointLabels = [
  'Shoulder pan',
  'Shoulder lift',
  'Elbow flex',
  'Wrist flex',
  'Wrist roll',
  'Gripper',
];

function TeleoperatePage() {
  return (
    <main className="teleoperate-page">
      <div className="teleoperate-heading">
        <h1>Teleoperate</h1>
        <p>Manipulate the followers through the leader's movements.</p>
      </div>

      <div className="teleoperate-layout">
        <section className="teleoperate-panel">
          <div className="teleoperate-panel-header">
            <h2>Camera Preview</h2>
            <span className="teleoperate-status">To be connected</span>
          </div>

          <div className="teleoperate-cameras">
            {['wrist', 'scene'].map((camera) => (
              <figure className="teleoperate-camera" key={camera}>
                <figcaption>{camera}</figcaption>
                <div className="teleoperate-camera-placeholder">
                  Live video link to be provided
                </div>
              </figure>
            ))}
          </div>
        </section>

        <div className="teleoperate-controls">
          <section className="teleoperate-panel">
            <div className="teleoperate-panel-header">
              <h2>Teleopertation Control</h2>
              <span className="teleoperate-status">Unconfirmed</span>
            </div>

            <dl className="teleoperate-details">
              <div>
                <dt>Leader</dt>
                <dd>Unconfirmed</dd>
              </div>
              <div>
                <dt>Follower</dt>
                <dd>Unconfirmed</dd>
              </div>
              <div>
                <dt>Calibration</dt>
                <dd>Unconfirmed</dd>
              </div>
            </dl>

            <div className="teleoperate-actions">
              <button type="button" className="teleoperate-start" disabled>
                Start
              </button>
              <button type="button" disabled>
                Stop
              </button>
            </div>

            <p className="teleoperate-note">Before connecting the robot.</p>
          </section>

          <section className="teleoperate-panel">
            <div className="teleoperate-panel-header">
              <h2>Joint condition</h2>
              <span className="teleoperate-status">Scheduled</span>
            </div>

            <table className="teleoperate-joints">
              <thead>
                <tr>
                  <th scope="col">Joint</th>
                  <th scope="col">Leader</th>
                  <th scope="col">Follower</th>
                </tr>
              </thead>
              <tbody>
                {jointLabels.map((joint) => (
                  <tr key={joint}>
                    <th scope="row">{joint}</th>
                    <td>—</td>
                    <td>—</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </div>
    </main>
  );
}

export default TeleoperatePage;
