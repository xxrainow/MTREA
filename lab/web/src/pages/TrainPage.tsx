import { useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { loadDatasets } from '../api/datasets';
import type { DatasetSummary } from '../api/datasets';
import { errorMessage } from '../api/hardware';
import { loadRemoteMarkers } from '../api/remote';
import type { RemoteMarker } from '../api/remote';
import { StartRefused, checkTraining, loadDescribe, loadRuns, loadTargets, startTraining } from '../api/training';
import type { Targets } from '../api/training';
import {
  canStart, displayValue, formatDuration, formGroups, isActiveState, requestKey, runNameFrom, toRequest,
} from '../training/model';
import type { CheckResult, Describe, Param, TrainingForm } from '../training/model';
import TrainRunMonitor from '../components/TrainRunMonitor';
import './TrainPage.css';

type Props = {
  form: TrainingForm;
  setForm: Dispatch<SetStateAction<TrainingForm>>;
  // The run on screen. Kept by App so menu navigation returns to it.
  jobId: string | null;
  onJobChange: (jobId: string | null) => void;
  onOpenCheckpoint: (id: string) => void;
  onOpenDatasets: () => void;
};

const markerLabels: Record<RemoteMarker, string> = {
  synced: 'synced', different: 'different copy on host', not_synced: 'not synced', not_sent_fake: 'fake', unknown: 'sync unknown',
};

function TrainPage({ form, setForm, jobId, onJobChange, onOpenCheckpoint, onOpenDatasets }: Props) {
  const [describe, setDescribe] = useState<Describe | null>(null);
  const [targets, setTargets] = useState<Targets | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [markers, setMarkers] = useState<Record<string, RemoteMarker> | null>(null);
  const [loadError, setLoadError] = useState('');
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checkedKey, setCheckedKey] = useState('');
  const [busy, setBusy] = useState<'check' | 'start' | null>(null);
  const [actionError, setActionError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    Promise.all([loadDescribe(signal), loadTargets(signal), loadDatasets(signal)])
      .then(([nextDescribe, nextTargets, nextDatasets]) => {
        setDescribe(nextDescribe);
        setTargets(nextTargets);
        setDatasets(nextDatasets);
      })
      .catch(reason => { if (!signal.aborted) setLoadError(errorMessage(reason)); });
    // Separate: it asks the GPU host and may be slow.
    loadRemoteMarkers(signal).then(setMarkers).catch(() => { /* markers are a hint */ });
    if (jobId === null) {
      // After a reload, return to a run that is still going.
      loadRuns(signal).then(runs => {
        const active = runs.find(r => isActiveState(r.state));
        if (active) onJobChange(active.job.id);
      }).catch(() => { /* the setup form still works */ });
    }
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once per visit
  }, []);

  const { request, errors } = useMemo(
    () => (describe ? toRequest(describe, form) : { request: null, errors: [] }), [describe, form]);
  const currentKey = requestKey(request);
  const startable = canStart(check, checkedKey, currentKey);

  function update(patch: Partial<TrainingForm>) {
    setForm(prev => {
      const next = { ...prev, ...patch };
      if (!next.runNameEdited && describe) {
        next.runName = runNameFrom(describe.run_name_template, String(next.values.method ?? ''),
          next.dataset?.name ?? '', String(next.values.seed ?? ''), new Date());
      }
      return next;
    });
    setActionError('');
  }
  function setValue(name: string, value: string | boolean) {
    update({ values: { ...form.values, [name]: value } });
  }

  async function runCheck() {
    if (!request) return;
    setBusy('check');
    setActionError('');
    try {
      setCheck(await checkTraining(request));
      setCheckedKey(currentKey);
    } catch (reason) {
      setCheck(null);
      setActionError(errorMessage(reason));
    } finally {
      setBusy(null);
    }
  }

  async function start() {
    if (!request || !startable) return;
    setBusy('start');
    setActionError('');
    try {
      const job = await startTraining(request);
      setCheck(null);
      setCheckedKey('');
      onJobChange(job.id);
    } catch (reason) {
      if (reason instanceof StartRefused && reason.check) setCheck(reason.check);
      setActionError(errorMessage(reason));
    } finally {
      setBusy(null);
    }
  }

  function newSetup() {
    onJobChange(null);
    // A new run needs a new name; keep the rest of the settings.
    update({ runNameEdited: false });
  }

  if (jobId) {
    return <main className="train-page">
      <div className="train-heading">
        <div><h1>Train</h1><p>Follow the run and its training progress.</p></div>
      </div>
      <TrainRunMonitor key={jobId} jobId={jobId} onNew={newSetup} onOpenCheckpoint={onOpenCheckpoint} />
    </main>;
  }

  const groups = describe ? formGroups(describe, form.fake) : null;
  const methodParam = describe?.params.find(p => p.name === 'method');
  const method = describe?.methods.find(m => m.name === form.values.method);
  const ssh = targets?.ssh;
  const datasetOptions = datasets.filter(d => form.target === 'local' || d.backend === 'real');
  const selectedKey = form.dataset ? `${form.dataset.backend}/${form.dataset.name}` : '';

  function field(p: Param) {
    const value = displayValue(p, form.values[p.name]);
    const required = p.required ? ' · Required' : '';
    if (p.type === 'bool') {
      return <label key={p.name} className="train-checkbox" title={p.help}>
        <input type="checkbox" checked={Boolean(value)} onChange={e => setValue(p.name, e.target.checked)} />{p.label}
      </label>;
    }
    if (p.choices) {
      return <label key={p.name} className="train-field" title={p.help}>{p.label}{required}
        <select value={String(value)} onChange={e => setValue(p.name, e.target.value)}>
          {p.required && <option value="">Select…</option>}
          {p.choices.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>;
    }
    const numeric = p.type === 'int' || p.type === 'float';
    return <label key={p.name} className="train-field" title={p.help}>{p.label}{required}
      <input type={numeric ? 'number' : 'text'} value={String(value)} step={p.type === 'float' ? 'any' : 1}
        min={p.min ?? undefined} placeholder={p.name === 'seed' ? 'e.g. 0' : undefined}
        onChange={e => setValue(p.name, e.target.value)} />
    </label>;
  }

  return <main className="train-page">
    <div className="train-heading">
      <div><h1>Train</h1><p>Choose a compute target and configure policy training.</p></div>
      <button type="button" className="train-button-primary" disabled={!startable || busy !== null}
        title={startable ? undefined : 'Check settings first; any change needs a new check.'}
        onClick={() => void start()}>{busy === 'start' ? 'Starting…' : 'Start training'}</button>
    </div>
    {loadError && <p className="train-errors" role="alert">{loadError}</p>}
    <section className="train-panel train-compute">
      <h2>Compute target</h2>
      <label className="train-field">Run training on
        <select value={form.target} onChange={e => update({ target: e.target.value as TrainingForm['target'], fake: e.target.value === 'local' ? true : form.fake, dataset: null })}>
          <option value="ssh">GPU host{ssh?.host ? ` · ${ssh.host}` : ''}</option>
          <option value="local">This machine (fake)</option>
        </select>
      </label>
      {form.target === 'ssh' && <>
        <label className="train-checkbox"><input type="checkbox" checked={form.fake} onChange={e => update({ fake: e.target.checked })} />Fake run on the GPU host (no GPU work; tests the remote path)</label>
        <p className="train-note">{!ssh ? 'Checking the GPU host…' : !ssh.configured ? `Not configured: ${ssh.message}` : ssh.reachable ? `Reachable · ${ssh.host}` : `Unreachable: ${ssh.message}`}</p>
      </>}
      {form.target === 'local' && <p className="train-note">This machine has no GPU for π0.5, so it runs fake training only: synthetic losses, placeholder checkpoints under data/fake/runs.</p>}
    </section>
    {groups && <div className="train-grid">
      <section className="train-panel"><h2>Training inputs</h2>
        <label className="train-field">Dataset
          <select value={selectedKey} onChange={e => {
            const picked = datasetOptions.find(d => `${d.backend}/${d.name}` === e.target.value);
            update({ dataset: picked ? { backend: picked.backend, name: picked.name } : null });
          }}>
            <option value="">Select a dataset</option>
            {datasetOptions.map(d => <option key={d.id} value={`${d.backend}/${d.name}`}>
              {d.name}{d.backend === 'fake' ? ' · fake' : ''}{form.target === 'ssh' && markers ? ` · ${markerLabels[markers[d.id] ?? 'unknown']}` : ''}
            </option>)}
          </select>
        </label>
        {form.target === 'ssh' && <p className="train-note">The GPU host trains on its synced copy. Send a dataset from Datasets → GPU server first. <button type="button" className="train-inline-link" onClick={onOpenDatasets}>Open Datasets</button></p>}
        {methodParam && field(methodParam)}
        {method && <p className="train-note">{method.description} Tested up to batch {method.max_batch_size_tested}{method.oom_batch_size ? `; out of memory at ${method.oom_batch_size}` : ''}.</p>}
        <label className="train-field">Run name
          <input value={form.runName} onChange={e => update({ runName: e.target.value, runNameEdited: true })} />
        </label>
        <p className="train-note">Run directory: {check && checkedKey === currentKey ? check.run_dir : `<runs on the ${form.target === 'ssh' ? 'GPU host' : 'this machine'}>/${form.runName || '<run name>'}`}</p>
      </section>
      <section className="train-panel"><h2>Training conditions</h2>
        <div className="train-field-pair">{groups.basic.filter(p => p.name !== 'method').map(field)}</div>
        <details><summary>Advanced settings</summary>
          <div className="train-field-pair">{groups.advanced.filter(p => p.type !== 'bool').map(field)}</div>
          {groups.advanced.filter(p => p.type === 'bool').map(field)}
        </details>
        {groups.fake.length > 0 && <details open><summary>Fake run</summary><div className="train-field-pair">{groups.fake.map(field)}</div></details>}
        <details><summary>Fixed for every run</summary>
          <dl className="train-details">
            {groups.fixed.map(p => <div key={p.name}><dt>{p.label}</dt><dd>{String(p.default)}</dd></div>)}
            {Object.entries(describe!.fixed.lerobot_flags).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
            <div><dt>LR schedule</dt><dd>{describe!.fixed.scheduler}</dd></div>
          </dl>
        </details>
      </section>
    </div>}
    <section className="train-panel train-checks"><h2>Pre-run checks</h2>
      {errors.length > 0 && <ul className="train-errors">{errors.map(e => <li key={e}>{e}</li>)}</ul>}
      {check && <>
        {checkedKey !== currentKey && <p className="train-note" role="status">Settings changed since this check. Check again before starting.</p>}
        <ul className="train-check-list">
          {check.checks.map(c => <li key={c.id} className={`train-check is-${c.status}`}>
            <span className="train-check-status">{c.status}</span>
            <span>{c.message}{c.link === 'datasets' && <> <button type="button" className="train-inline-link" onClick={onOpenDatasets}>Open Datasets</button></>}</span>
          </li>)}
        </ul>
        <dl className="train-details">
          <div><dt>Result</dt><dd role="status">{check.ok ? 'Ready to start' : 'Not ready'}</dd></div>
          <div><dt>Estimated time</dt><dd>{formatDuration(check.estimated_seconds)}{check.estimated_seconds != null ? ' of training steps' : ''}</dd></div>
          {check.host && <div><dt>Host</dt><dd>{check.host}</dd></div>}
        </dl>
        {check.command && <details><summary>Command</summary><pre className="train-log">{check.command}</pre></details>}
      </>}
      {actionError && <p className="train-errors" role="alert">{actionError}</p>}
      <div className="train-actions">
        <button type="button" disabled={!request || busy !== null} onClick={() => void runCheck()}>{busy === 'check' ? 'Checking on the target…' : 'Check settings'}</button>
      </div>
      <p className="train-note">Checks run on the target machine with train.sh --check. Start is enabled only after a passing check with unchanged settings.</p>
    </section>
  </main>;
}
export default TrainPage;
