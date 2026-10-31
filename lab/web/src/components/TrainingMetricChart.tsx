import type { Metric } from '../training/model';
import '../pages/TrainPage.css';

type Kind = 'loss' | 'lr' | 'grad_norm';
type Props = { metrics: Metric[]; kind: Kind; fake?: boolean };
const titles: Record<Kind, string> = { loss: 'Loss', lr: 'Learning rate', grad_norm: 'Gradient norm' };

export default function TrainingMetricChart({ metrics, kind, fake = false }: Props) {
  const title = titles[kind];
  const format = (value: number) => kind === 'lr' ? value.toExponential(1) : value.toFixed(3);
  const points = metrics.filter((m): m is Metric & Record<Kind, number> => typeof m[kind] === 'number');
  const last = points.at(-1);
  const values = points.map(m => m[kind]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * .1, Math.abs(high) * .05, 1e-9);
  const min = Math.max(0, low - padding);
  const max = high + padding;
  const firstStep = points[0]?.step ?? 0;
  const lastStep = last?.step ?? 1;
  const x = (step: number) => 72 + (step - firstStep) / Math.max(1, lastStep - firstStep) * 400;
  const y = (value: number) => 190 - (value - min) / (max - min) * 160;
  return <section className="train-panel train-metric-panel">
    <div className="train-panel-heading"><h2>{title}</h2><span>{last ? format(last[kind]) : '—'}</span></div>
    <p className="train-note">{fake ? 'Fake run · synthetic values' : 'From metrics.jsonl'} · window averages as lerobot logs them</p>
    {!last ? <div className="train-metric-empty">Waiting for the first logged step…</div> : <svg viewBox="0 0 500 240" role="img" aria-label={`${title} by training step. Latest step ${last.step}: ${format(last[kind])}`}>
      {[0, .5, 1].map(t => { const value = min + (max - min) * t; return <g key={t}><line x1="72" x2="472" y1={y(value)} y2={y(value)} className="train-chart-grid" /><text x="64" y={y(value) + 4} textAnchor="end">{format(value)}</text></g>; })}
      <polyline points={points.map(m => `${x(m.step)},${y(m[kind])}`).join(' ')} className="train-chart-line" />
      <circle cx={x(last.step)} cy={y(last[kind])} r="3" className="train-chart-point" />
      <text x="72" y="211">{firstStep.toLocaleString()}</text><text x="472" y="211" textAnchor="end">{lastStep === firstStep ? '' : lastStep.toLocaleString()}</text><text x="272" y="232" textAnchor="middle">Training step</text>
    </svg>}
  </section>;
}
