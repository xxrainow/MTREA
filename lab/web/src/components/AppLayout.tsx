import { useState } from 'react';
import type { ReactNode } from 'react';
import './AppLayout.css';

export const menuItems = [
  'Overview',
  'Devices',
  'Teleoperate',
  'Collect',
  'Datasets',
  'Train',
  'Run & Evaluate',
  'Jobs',
] as const;

export type PageName = (typeof menuItems)[number];

type AppLayoutProps = {
  children: ReactNode;
  activePage: PageName;
  onPageChange: (page: PageName) => void;
};

function AppLayout({ children, activePage, onPageChange }: AppLayoutProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  return (
    <div className="app-layout" data-theme={theme}>
      <aside className="app-sidebar">
        <div className="app-brand">
          <span className="app-brand-mark">M</span>
          <div>
            <strong>MTREA</strong>
            <span className="app-brand-subtitle">ROBOT LEARNING LAB</span>
          </div>
        </div>

        <div className="app-workspace">SO-101 Workspace</div>

        <nav className="app-nav" aria-label="menu">
          {menuItems.map((item) => (
            <button
              key={item}
              type="button"
              className={item === activePage ? 'is-active' : ''}
              aria-current={item === activePage ? 'page' : undefined}
              onClick={() => onPageChange(item)}
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <div className="app-body">
        <header className="app-header">
          <span>MTREA LAB / {activePage}</span>

          <div className="app-header-actions">
            <div
              className="app-connection-status"
              aria-label="Connection status"
            >
              {['Robot', 'GPU', 'Policy'].map((label) => (
                <span className="app-connection-item" key={label}>
                  <span className="app-connection-dot" aria-hidden="true" />
                  <span>{label}</span>
                  <span className="app-connection-value">Not checked</span>
                </span>
              ))}
            </div>
            <button
              type="button"
              className="app-theme-button"
              onClick={() =>
                setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
              }
            >
              {theme === 'dark' ? '☼' : '☽'}
            </button>
          </div>
        </header>

        <div className="app-content">{children}</div>
      </div>
    </div>
  );
}

export default AppLayout;
