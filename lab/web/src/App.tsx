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
import CheckpointPage from './pages/CheckpointPage';
import { initialForm } from './training/model';
import type { TrainingForm } from './training/model';
import type { DatasetReference } from './api/datasets';

function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview');
  const [collectBusy, setCollectBusy] = useState(false);
  const [datasetToOpen, setDatasetToOpen] = useState<DatasetReference | null>(null);
  // Train's draft settings and the run it shows survive menu navigation; the
  // run itself lives on the backend, so a reload finds it again.
  const [trainForm, setTrainForm] = useState<TrainingForm>(initialForm);
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [checkpointToOpen, setCheckpointToOpen] = useState<string | null>(null);
  const [jobToOpen, setJobToOpen] = useState<string | null>(null);

  function openCheckpoint(id: string | null) {
    setCheckpointToOpen(id);
    setActivePage('Checkpoint');
  }

  function navigate(page: PageName) {
    if (activePage === 'Collect' && page !== 'Collect' && collectBusy) {
      window.alert('Recording is active. Stop recording and wait for saving to finish before leaving.');
      return;
    }

    setCheckpointToOpen(null);
    setDatasetToOpen(null);
    setJobToOpen(null);
    setActivePage(page);
  }

  function openJob(jobId: string) {
    setJobToOpen(jobId);
    setActivePage('Jobs');
  }

  function openDataset(dataset: DatasetReference) {
    setDatasetToOpen(dataset);
    setActivePage('Datasets');
  }

  return (
    <AppLayout activePage={activePage} onPageChange={navigate}>
      {activePage === 'Overview' ? (
        <OverviewPage onNavigate={navigate} />
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
          onOpenJob={openJob}
        />
      ) : activePage === 'Train' ? (
        <TrainPage
          form={trainForm}
          setForm={setTrainForm}
          jobId={trainJobId}
          onJobChange={setTrainJobId}
          onOpenCheckpoint={openCheckpoint}
          onOpenDatasets={() => navigate('Datasets')}
        />
      ) : activePage === 'Checkpoint' ? (
        <CheckpointPage selectedId={checkpointToOpen} onSelect={setCheckpointToOpen} />
      ) : activePage === 'Run & Evaluate' ? (
        <RunPage />
      ) : activePage === 'Jobs' ? (
        <JobsPage key={jobToOpen ?? 'all'} initialJobId={jobToOpen} />
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
