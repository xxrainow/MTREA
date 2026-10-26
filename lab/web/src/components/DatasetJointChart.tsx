import { useEffect, useState } from 'react';
import { loadEpisodeJoints } from '../api/datasets';
import type {
  DatasetSummary,
  EpisodeJoints,
} from '../api/datasets';

type Props = {
  dataset: DatasetSummary;
  index: number;
};

function JointLine({
  data,
  jointIndex,
}: {
  data: EpisodeJoints;
  jointIndex: number;
}) {
  const values = data.positions.map(
    (row) => row[jointIndex] ?? null,
  );
  const validValues = values.filter(
    (value): value is number =>
      value !== null && Number.isFinite(value),
  );

  if (data.timestamps.length === 0 || validValues.length === 0) {
    return <p className="datasets-note">No valid joint samples.</p>;
  }

  const minimum = validValues.reduce(
    (result, value) => Math.min(result, value),
    Infinity,
  );
  const maximum = validValues.reduce(
    (result, value) => Math.max(result, value),
    -Infinity,
  );

  const padding = (maximum - minimum) * 0.05 || 1;
  const lower = minimum - padding;
  const upper = maximum + padding;

  const start = data.timestamps[0];
  const duration = data.timestamps[data.timestamps.length - 1] - start;

  const x = (timestamp: number) =>
    65 + ((timestamp - start) / (duration || 1)) * 810;

  const y = (value: number) =>
    20 + ((upper - value) / (upper - lower)) * 160;

  let path = '';
  let move = true;

  values.forEach((value, sampleIndex) => {
    if (value === null || !Number.isFinite(value)) {
      move = true;
      return;
    }

    const timestamp = data.timestamps[sampleIndex];
    path += `${move ? 'M' : 'L'} ${x(timestamp)} ${y(value)} `;
    move = false;
  });

  return (
    <div className="datasets-joint-chart-scroll">
      <svg
        className="datasets-joint-chart"
        viewBox="0 0 900 230"
        role="img"
        aria-label={`${data.joint_names[jointIndex]} recorded positions over time`}
      >
        <line x1="65" y1="20" x2="65" y2="180" stroke="currentColor" />
        <line x1="65" y1="180" x2="875" y2="180" stroke="currentColor" />

        <text x="55" y="25" textAnchor="end">
          {upper.toFixed(2)}
        </text>
        <text x="55" y="180" textAnchor="end">
          {lower.toFixed(2)}
        </text>

        <path
          d={path}
          fill="none"
          stroke="var(--lab-accent)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        <text x="65" y="202">0</text>
        <text x="875" y="202" textAnchor="end">
          {duration.toFixed(2)}
        </text>
        <text x="470" y="223" textAnchor="middle">
          Time (s)
        </text>
      </svg>
    </div>
  );
}

export default function DatasetJointChart({ dataset, index }: Props) {
  const [data, setData] = useState<EpisodeJoints | null>(null);
  const [jointIndex, setJointIndex] = useState(0);
  const [error, setError] = useState('');

  useEffect(() => {
    if (dataset.backend === 'fake') return;

    const controller = new AbortController();

    loadEpisodeJoints(dataset, index, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setData(result);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            reason instanceof Error
              ? reason.message
              : 'Cannot load joint data.',
          );
        }
      });

    return () => controller.abort();
  }, [dataset, index]);

  return (
    <section className="datasets-joint-panel" aria-label="Joint positions">
      <div className="datasets-panel-header">
        <h3>Joint positions</h3>

        {data && (
          <label className="datasets-field">
            Joint
            <select
              value={jointIndex}
              onChange={(event) =>
                setJointIndex(Number(event.target.value))
              }
            >
              {data.joint_names.map((name, value) => (
                <option key={name} value={value}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {dataset.backend === 'fake' ? (
        <p className="datasets-note">
          Joint measurements are not recorded in Fake mode.
        </p>
      ) : error ? (
        <p className="datasets-error" role="alert">{error}</p>
      ) : data ? (
        <>
          <JointLine data={data} jointIndex={jointIndex} />
          <p className="datasets-note">
            {data.timestamps.length} recorded samples · Stored values;
            units depend on the recording configuration.
          </p>
        </>
      ) : (
        <p role="status">Loading joint data…</p>
      )}
    </section>
  );
}
