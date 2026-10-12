import { useState } from 'react';
import AppLayout from './components/AppLayout';
import type { PageName } from './components/AppLayout';
import JobsPage from './pages/JobsPage';
import OverviewPage from './pages/OverviewPage';
import DevicesPage from './pages/DevicesPage';
import TeleoperatePage from './pages/TeleoperatePage';
import CollectPage from './pages/CollectPage';
import DatasetsPage from './pages/DatasetsPage';
import TrainPage from './pages/TrainPage';
import RunPage from './pages/RunPage';

function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview');

  return (
    <AppLayout activePage={activePage} onPageChange={setActivePage}>
      {activePage === 'Overview' ? (
        <OverviewPage onNavigate={setActivePage} />
      ) : activePage === 'Devices' ? (
        <DevicesPage />
      ) : activePage === 'Teleoperate' ? (
        <TeleoperatePage />
      ) : activePage === 'Collect' ? (
        <CollectPage />
      ) : activePage === 'Datasets' ? (
        <DatasetsPage />
      ) : activePage === 'Train' ? (
        <TrainPage />
      ) : activePage === 'Run & Evaluate' ? (
        <RunPage />
      ) : activePage === 'Jobs' ? (
        <JobsPage />
      ) : (
        <section>
          <h1>{activePage}</h1>
          <p>This screen is being prepared.</p>
        </section>
      )}
    </AppLayout>
  );
}

export default App;
