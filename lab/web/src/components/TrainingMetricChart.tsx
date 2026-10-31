import type { TrainingMetric } from '../training/model';
import '../pages/TrainPage.css';

type Props = { metrics: TrainingMetric[]; kind: 'loss' | 'lr' };
export default function TrainingMetricChart({ metrics, kind }: Props) {
  const title = kind === 'loss' ? 'Loss' : 'Learning rate';
  const format = (value: number) => kind === 'lr' ? value.toExponential(1) : value.toFixed(3);
  const last = metrics.at(-1);
  const values = metrics.map(m => m[kind]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * .1, Math.abs(high) * .05, 1e-9);
  const min = Math.max(0, low - padding);
  const max = high + padding;
  const firstStep = metrics[0]?.step ?? 0;
  const lastStep = last?.step ?? 1;
  const x = (step: number) => 72 + (step - firstStep) / Math.max(1, lastStep - firstStep) * 400;
  const y = (value: number) => 190 - (value - min) / (max - min) * 160;
  return <section className="train-panel train-metric-panel">
    <div className="train-panel-heading"><h2>{title}</h2><span>{last ? format(last[kind]) : '—'}</span></div>
    <p className="train-note">Synthetic demo values · Latest 500 samples</p>
    {!last ? <div className="train-metric-empty">Waiting for first metric…</div> : <svg viewBox="0 0 500 240" role="img" aria-label={`${title} by training step, simulated. Latest step ${last.step}: ${format(last[kind])}`}>
      {[0, .5, 1].map(t => { const value = min + (max - min) * t; return <g key={t}><line x1="72" x2="472" y1={y(value)} y2={y(value)} className="train-chart-grid" /><text x="64" y={y(value) + 4} textAnchor="end">{format(value)}</text></g>; })}
      <polyline points={metrics.map(m => `${x(m.step)},${y(m[kind])}`).join(' ')} className="train-chart-line" />
      <circle cx={x(last.step)} cy={y(last[kind])} r="3" className="train-chart-point" />
      <text x="72" y="211">{firstStep.toLocaleString()}</text><text x="472" y="211" textAnchor="end">{lastStep === firstStep ? '' : lastStep.toLocaleString()}</text><text x="272" y="232" textAnchor="middle">Training step</text>
    </svg>}
  </section>;
}
