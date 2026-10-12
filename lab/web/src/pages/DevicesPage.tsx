import './DevicesPage.css';

const arms = [
  { name: 'Follower', description: 'Robot arm that follows commands' },
  { name: 'Leader', description: 'Input robot arm controlled by the operator' },
];

function DevicesPage() {
  return (
    <main className="devices-page">
      <div className="devices-heading">
        <h1>Devices</h1>
        <p>Manage robot connections, calibration, and cameras.</p>
      </div>

      <div className="devices-grid">
        {arms.map((arm) => (
          <section className="device-panel" key={arm.name}>
            <div className="device-panel-header">
              <h2>{arm.name}</h2>
              <span className="device-status">Not checked</span>
            </div>

            <p className="device-description">{arm.description}</p>

            <dl className="device-details">
              <div>
                <dt>Robot type</dt>
                <dd>SO-101</dd>
              </div>
              <div>
                <dt>Device ID</dt>
                <dd>Configuration pending</dd>
              </div>
              <div>
                <dt>Serial port</dt>
                <dd>Configuration pending</dd>
              </div>
              <div>
                <dt>Calibration</dt>
                <dd>File verification pending</dd>
              </div>
            </dl>

            <div className="device-actions">
              <button type="button" disabled>
                Connect port
              </button>
              <button type="button" disabled>
                Calibrate
              </button>
            </div>
          </section>
        ))}
      </div>

      <div className="devices-section-heading">
        <h2>Cameras</h2>
        <span className="device-description">Preview integration pending</span>
      </div>

      <div className="devices-grid">
        {['wrist', 'scene'].map((camera) => (
          <section className="device-panel" key={camera}>
            <div className="device-panel-header">
              <h2>{camera}</h2>
              <span className="device-status">Not checked</span>
            </div>

            <div
              className="device-camera-placeholder"
              aria-label={`${camera} camera preview not connected`}
            >
              Camera preview
            </div>

            <button type="button" disabled>
              Check camera
            </button>
          </section>
        ))}
      </div>
    </main>
  );
}

export default DevicesPage;
