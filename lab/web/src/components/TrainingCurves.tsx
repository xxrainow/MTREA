import type { Metric } from '../training/model';
import TrainingMetricChart from './TrainingMetricChart';

type Props = { metrics: Metric[]; fake?: boolean; emptyMessage?: string };

export default function TrainingCurves({ metrics, fake, emptyMessage }: Props) {
  return <div className="train-grid train-grid-3">
    <TrainingMetricChart metrics={metrics} kind="loss" fake={fake} emptyMessage={emptyMessage} />
    <TrainingMetricChart metrics={metrics} kind="lr" fake={fake} emptyMessage={emptyMessage} />
    <TrainingMetricChart metrics={metrics} kind="grad_norm" fake={fake} emptyMessage={emptyMessage} />
  </div>;
}
