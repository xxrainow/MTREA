import type { PageName } from '../components/AppLayout';
import './OverviewPage.css';

type OverviewPageProps = {
  onNavigate: (page: PageName) => void;
};

function OverviewPage({ onNavigate }: OverviewPageProps) {
  return (
    <main className="overview-page">
      <div className="overview-heading">
        <h1>Overview</h1>
        <p>Prepare your robots, collect data, and train policies.</p>
      </div>

      <div className="overview-panels">
        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Robot</h2>
          </div>

          <p className="overview-value">SO-101</p>
          <p className="overview-description">Leader · Follower · Camera</p>

          <div className="overview-actions">
            <button type="button" onClick={() => onNavigate('Devices')}>
              Device Configuration
            </button>

            <button
              type="button"
              className="overview-button-accent"
              onClick={() => onNavigate('Teleoperate')}
            >
              Teleoperation
            </button>
          </div>
        </section>

        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Dataset</h2>
          </div>

          <p className="overview-value">Dataset</p>
          <p className="overview-description">
            Recording Episode · Episode Navigation · Data Inspection
          </p>

          <div className="overview-actions">
            <button type="button" onClick={() => onNavigate('Datasets')}>
              Datasets
            </button>

            <button type="button" onClick={() => onNavigate('Collect')}>
              Collect episodes
            </button>
          </div>
        </section>

        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Create a policy</h2>
          </div>

          <p className="overview-value">π0.5</p>
          <p className="overview-description">H0 · Full fine-tuning</p>

          <div className="overview-actions">
            <button
              type="button"
              className="overview-button-green"
              onClick={() => onNavigate('Train')}
            >
              Training
            </button>

            <button type="button" onClick={() => onNavigate('Run & Evaluate')}>
              Run & Evaluate
            </button>
          </div>
        </section>
      </div>

      <section className="overview-jobs">
        <div className="overview-jobs-header">
          <h2>Jobs</h2>
          <button type="button" onClick={() => onNavigate('Jobs')}>
            Go to Job Page →
          </button>
        </div>

        <p>No active jobs. Start one from the Training page.</p>
      </section>
    </main>
  );
}

export default OverviewPage;
