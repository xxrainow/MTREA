import { useEffect, useReducer, useState } from 'react';
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
import { initialTrainingState, trainingReducer } from './training/model';
import type { DatasetReference } from './api/datasets';

function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview');
  const [collectBusy, setCollectBusy] = useState(false);
  const [datasetToOpen, setDatasetToOpen] = useState<DatasetReference | null>(null);
  const [training, dispatchTraining] = useReducer(trainingReducer, initialTrainingState);
  const [checkpointToOpen, setCheckpointToOpen] = useState<string | null>(null);

  useEffect(() => {
    const run = training.run;
    if (run?.status !== 'running') return;
    const timer = window.setInterval(() => {
      dispatchTraining({ type: 'tick' });

    }, 1000);
    return () => window.clearInterval(timer);
  }, [training.run]);

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
    setActivePage(page);
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
        />
      ) : activePage === 'Train' ? (
        <TrainPage state={training} dispatch={dispatchTraining} onOpenCheckpoint={openCheckpoint} />
      ) : activePage === 'Checkpoint' ? (
        <CheckpointPage runs={training.runs} checkpoints={training.checkpoints} selectedId={checkpointToOpen} onSelect={openCheckpoint} />
      ) : activePage === 'Run & Evaluate' ? (
        <RunPage />
      ) : activePage === 'Jobs' ? (
        <JobsPage trainingRuns={training.runs} />
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
