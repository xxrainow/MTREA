import { useRef, useState } from 'react';
import AppLayout from '../components/AppLayout';
import '../index.css';
import '../pages/DatasetsPage.css';
import '../components/DatasetMergeDialog.css';

// All data and compatibility results on this standalone page are illustrative.
const samples = [
  { name: 'pickplace_blocks_session01', episodes: 24, frames: 10800, note: 'Same recording format', available: true },
  { name: 'pickplace_blocks_session02', episodes: 18, frames: 8100, note: 'Same recording format', available: true },
  { name: 'pickplace_blocks_session03', episodes: 12, frames: 5400, note: 'Same recording format', available: true },
  { name: 'pickplace_single_camera', episodes: 8, frames: 3600, note: 'Camera features differ', available: false },
];

export default function MergePreview() {
  const modal = useRef<HTMLDialogElement>(null);
  const [selected, setSelected] = useState([0]);
  const [search, setSearch] = useState('');
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const [step, setStep] = useState<'setup' | 'review' | 'done'>('setup');
  const [catalog, setCatalog] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [notice, setNotice] = useState('');
  const sources = selected.map((index) => samples[index]);
  const episodes = sources.reduce((sum, item) => sum + item.episodes, 0);
  const frames = sources.reduce((sum, item) => sum + item.frames, 0);
  const nameError = !name ? '' : !/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(name) ? 'Use letters, numbers, underscores or hyphens.' : samples.some((item) => item.name === name) ? 'This name already exists. Choose a new name.' : '';
  const ready = selected.length > 1 && name.length > 0 && !nameError;

  function openMerge() {
    setSelected([0]); setName(''); setStep('setup'); setSearch(''); setAdding(false);
    modal.current?.showModal();
  }

  return <AppLayout activePage="Datasets" onPageChange={() => setNotice('This preview focuses on Datasets. Other pages are available in the main app.')}>
    <main className="datasets-page merge-preview">
      <div className="merge-preview-banner"><span><strong>DESIGN PREVIEW</strong> · Sample datasets · No files are written</span><span>Datasets / Merge</span></div>
      {notice && <p className="datasets-notice" role="status">{notice}</p>}
      {!catalog && <button className="datasets-back-button" onClick={() => setCatalog(true)}>← Dataset list</button>}
      <div className="datasets-heading">
        <div><h1>{catalog ? 'Datasets' : samples[0].name}</h1><p>{catalog ? 'Browse locally collected demonstrations and metadata.' : 'Saved recording metadata and episode instructions'}</p></div>
        <div className="merge-actions"><span className="datasets-mode is-real">REAL · SAMPLE</span><button className="datasets-button merge-primary" onClick={openMerge}>⑂ Merge datasets</button><button className="datasets-button datasets-delete-button" onClick={() => { if (window.confirm("Preview: archive this dataset? No files will be changed.")) setNotice("Delete preview complete. No files were changed."); }}>Delete</button></div>
      </div>
      {catalog ? <section className="datasets-panel"><div className="datasets-panel-header"><h2>Collected datasets</h2><span className="datasets-status">4 sample datasets</span></div><div className="datasets-table-wrapper"><table className="datasets-table"><thead><tr><th>Dataset</th><th>Mode</th><th>Episodes</th><th>Frames</th></tr></thead><tbody>{samples.map((item) => <tr key={item.name}><td>{item === samples[0] ? <button className="datasets-name-button" onClick={() => setCatalog(false)}>{item.name}</button> : item.name}</td><td>REAL</td><td>{item.episodes}</td><td>{item.frames.toLocaleString()}</td></tr>)}</tbody></table></div></section> : <>
        <dl className="datasets-summary"><div><dt>Episodes</dt><dd>24</dd></div><div><dt>Total frames</dt><dd>10,800</dd></div><div><dt>Recording format</dt><dd>Illustrative compatible sources</dd></div></dl>
        <section className="datasets-panel datasets-remote"><div className="datasets-panel-header"><h2>GPU server</h2><span className="datasets-status">Not synced · sample</span></div><p className="datasets-description">Merge locally, inspect the new dataset, then sync it as a separate dataset.</p></section>
        <div className="datasets-browser"><section className="datasets-panel datasets-episodes"><div className="datasets-panel-header"><h2>Episodes</h2><span className="datasets-status">24</span></div><ul className="datasets-episode-list">{Array.from({ length: 24 }, (_, index) => <li key={index}><button className={`datasets-episode-button ${episode === index ? 'is-active' : ''}`} onClick={() => setEpisode(index)}><span className="datasets-episode-index">{index}</span><span>Episode {index}</span></button></li>)}</ul></section>
          <section className="datasets-panel datasets-viewer"><div className="datasets-panel-header"><h2>Episode {episode}</h2><span className="datasets-status">Sample metadata</span></div><div className="merge-video-placeholder"><span className="merge-placeholder-icon">▷</span><strong>Episode video</strong><span>Recordings appear here in the actual dataset.</span><small>No recorded video is loaded in this preview.</small></div><dl className="datasets-details"><div><dt>Instruction</dt><dd>Pick up the block and place it in the tray.</dd></div><div><dt>Storage location</dt><dd>data/raw/{samples[0].name}</dd></div></dl></section>
        </div>
      </>}
      <dialog ref={modal} className="merge-dialog" aria-labelledby="merge-title">
        <div className="merge-dialog-heading"><div><span className="merge-eyebrow">LOCAL DATASETS <span> / PREVIEW</span></span><h2 id="merge-title">{step === 'setup' ? 'Merge datasets' : step === 'review' ? 'Review merge' : 'Merge preview complete'}</h2></div><button className="datasets-button" aria-label="Close merge" onClick={() => modal.current?.close()}>×</button></div>
        {step === 'setup' ? <>
          <p className="merge-description">Combine saved episodes into a new local dataset. Source datasets stay unchanged.</p>
          <div className="merge-section-label"><h3>Source datasets</h3><span>{selected.length} selected</span></div>
          <div className="merge-source-list">{selected.map((index) => <div className="merge-source" key={index}><span className="merge-source-icon">▤</span><div><strong>{samples[index].name}</strong><small>{samples[index].episodes} episodes · {samples[index].frames.toLocaleString()} frames · Real</small></div>{index === 0 ? <span className="merge-chip">Current</span> : <button className="merge-remove" aria-label={`Remove ${samples[index].name}`} onClick={() => setSelected(selected.filter((value) => value !== index))}>×</button>}</div>)}</div>
          <button className="datasets-button merge-add" aria-expanded={adding} onClick={() => setAdding(!adding)}>+ Add dataset</button>
          {adding && <div className="merge-picker"><input autoFocus aria-label="Search datasets" placeholder="Search local datasets…" value={search} onChange={(event) => setSearch(event.target.value)} /><div className="merge-picker-results">{samples.filter((item, index) => !selected.includes(index) && item.name.includes(search.toLowerCase())).map((item) => <button key={item.name} disabled={!item.available} onClick={() => { setSelected([...selected, samples.indexOf(item)]); setAdding(false); setSearch(''); }}><span>{item.name}<small>{item.episodes} episodes · {item.note}</small></span><span>{item.available ? '+' : 'Unavailable'}</span></button>)}{!samples.some((item, index) => !selected.includes(index) && item.name.includes(search.toLowerCase())) && <p>No matching datasets.</p>}</div><small>Real sources only. Compatibility labels are simulated.</small></div>}
          {selected.length < 2 && <p className="merge-hint">Add at least one more dataset to continue.</p>}
          <label className="merge-name">Merged dataset name<input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. pickplace_blocks_combined" aria-describedby="merge-name-help" aria-invalid={!!nameError} /></label>
          <p id="merge-name-help" className={nameError ? 'merge-error' : 'merge-hint'}>{nameError || `New local dataset: data/raw/${name || '<name>'}`}</p>
          <div className="merge-total"><div><strong>{episodes}</strong><span>Total episodes</span></div><div><strong>{frames.toLocaleString()}</strong><span>Total frames</span></div><div><strong>{selected.length}</strong><span>Source datasets</span></div></div>
          <p className="merge-hint">Research roles are not inferred. Keep oracle and held-out demos out of training datasets for compared methods.</p>
          <footer><button className="datasets-button" onClick={() => modal.current?.close()}>Cancel</button><button className="datasets-button merge-primary" disabled={!ready} onClick={() => { setAdding(false); setStep('review'); }}>Review merge →</button></footer>
        </> : step === 'review' ? <>
          <p className="merge-description">Check the proposed output before creating a new dataset.</p>
          <div className="merge-review-output"><span className="merge-eyebrow">NEW DATASET</span><h3>{name}</h3><p>{episodes} episodes · {frames.toLocaleString()} frames</p><code>data/raw/{name}</code></div>
          <h3 className="merge-review-label">Included sources</h3>{sources.map((item) => <div className="merge-review-row" key={item.name}><span>{item.name}</span><span>{item.episodes} episodes</span></div>)}
          <p className="merge-description">The real workflow will check dataset format, features and recording settings before merging, and retain source provenance in the output.</p>
          <p className="merge-hint">This preview simulates the result only. No compatibility check or file operation has run.</p>
          <footer><button className="datasets-button" onClick={() => setStep('setup')}>← Back</button><button className="datasets-button merge-primary" onClick={() => setStep('done')}>Simulate merge</button></footer>
        </> : <>
          <div className="merge-success"><span>✓</span><h3>{name}</h3><p>{episodes} episodes from {selected.length} source datasets</p></div><p className="merge-description">The proposed dataset is ready to review. This is a simulated completion; no dataset was created.</p>
          <footer><button className="datasets-button" onClick={() => setStep('setup')}>Edit selection</button><button className="datasets-button merge-primary" onClick={() => modal.current?.close()}>Done</button></footer>
        </>}
      </dialog>
    </main>
  </AppLayout>;
}

