import { useState } from 'react';
import AppLayout from './components/AppLayout';
import type { PageName } from './components/AppLayout';
import JobsPage from './pages/JobsPage';
import OverviewPage from './pages/OverviewPage';
import DevicesPage from './pages/DevicesPage';

function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview');

  return (
    <AppLayout activePage={activePage} onPageChange={setActivePage}>
    
      {activePage === 'Overview' ? (
        <OverviewPage onNavigate={setActivePage} />
      ) : activePage === 'Devices' ? (
        <DevicesPage />
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
