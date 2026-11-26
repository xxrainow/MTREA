import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import { loadMetrics } from '../api/training';
import { appendMetrics } from '../training/model';
import type { Metric } from '../training/model';
import TrainingCurves from './TrainingCurves';

type Props = { jobId: string; fake: boolean };

export default function CheckpointTrainingCurves({ jobId, fake }: Props) {
  const [metrics, setMetrics] = useState<Metric[] | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    loadMetrics(jobId, 0, controller.signal)
      .then(page => {
        if (!controller.signal.aborted) setMetrics(appendMetrics([], page.metrics, Infinity));
      })
      .catch(reason => {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      });
    return () => controller.abort();
  }, [jobId]);

  return <section className="checkpoint-training-curves" aria-label="Training curves">
    <h2>Training curves</h2>
    <p className="train-note">Full training history for this run, including when viewing a kept checkpoint.</p>
    {error ? <p className="train-errors" role="alert">Cannot load training curves: {error}. Use Refresh to retry.</p>
      : metrics === null ? <p className="train-note" role="status">Loading training curves…</p>
        : <TrainingCurves metrics={metrics} fake={fake} emptyMessage="No recorded values for this metric." />}
  </section>;
}
