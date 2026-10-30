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
import type { DatasetReference } from './api/datasets';

function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview');
  const [collectBusy, setCollectBusy] = useState(false);
  const [datasetToOpen, setDatasetToOpen] = useState<DatasetReference | null>(null);
  function navigate(page: PageName) {
    if (activePage === 'Collect' && page !== 'Collect' && collectBusy) {
      window.alert('Recording is active. Stop recording and wait for saving to finish before leaving.');
      return;
    }

    setDatasetToOpen(null);
    setActivePage(page);
  }

  function openDataset(dataset: DatasetReference) {
    setDatasetToOpen(dataset);
    setActivePage('Datasets');
  }

  return (
    <AppLayout activePage={activePage} onPageChange={navigate}>
      {activePage === 'Overview' ? (
        <OverviewPage onNavigate={setActivePage} />
      ) : activePage === 'Devices' ? (
        <DevicesPage />
      ) : activePage === 'Teleoperate' ? (
        <TeleoperatePage />
      ) : activePage === 'Collect' ? (
        <CollectPage onBusyChange={setCollectBusy} onOpenDataset={openDataset} />
      ) : activePage === 'Datasets' ? (
        <DatasetsPage
          key={datasetToOpen ? `${datasetToOpen.backend}:${datasetToOpen.name}` : 'list'}
          initialDataset={datasetToOpen}
        />
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
