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

const JOINT_COLORS = ['#ff9b43', '#6ea8ff', '#53cf91', '#bb93f9', '#ead45e', '#ff737e'];

function jointColor(index: number) {
  return JOINT_COLORS[index % JOINT_COLORS.length];
}

export function JointLines({
  data,
  jointIndex,
}: {
  data: EpisodeJoints;
  jointIndex: number | null;
}) {
  const indices = data.joint_names
    .map((_, index) => index)
    .filter((index) => jointIndex === null || index === jointIndex);
  let minimum = Infinity;
  let maximum = -Infinity;

  data.positions.forEach((row, sampleIndex) => {
    if (!Number.isFinite(data.timestamps[sampleIndex])) return;
    indices.forEach((index) => {
      const value = row[index];
      if (typeof value === 'number' && Number.isFinite(value)) {
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    });
  });

  if (data.timestamps.length === 0 || !Number.isFinite(minimum)) {
    return <p className="datasets-note">No valid joint samples.</p>;
  }

  const padding = (maximum - minimum) * 0.05 || 1;
  const lower = minimum - padding;
  const upper = maximum + padding;
  const start = data.timestamps[0];
  const duration = data.timestamps[data.timestamps.length - 1] - start;
  const x = (timestamp: number) => 65 + ((timestamp - start) / (duration || 1)) * 810;
  const y = (value: number) => 15 + ((upper - value) / (upper - lower)) * 90;

  return (
    <div className="datasets-joint-chart-scroll">
      <svg
        className="datasets-joint-chart"
        viewBox="0 0 900 150"
        role="img"
        aria-label={`${jointIndex === null ? 'All joints' : data.joint_names[jointIndex]} recorded positions over time`}
      >
        {[0, 0.5, 1].map((fraction) => {
          const value = lower + (upper - lower) * fraction;
          return (
            <g key={fraction}>
              <line x1="65" y1={y(value)} x2="875" y2={y(value)} stroke="currentColor" opacity="0.2" />
              <text x="55" y={y(value) + 4} textAnchor="end">{value.toFixed(2)}</text>
            </g>
          );
        })}
        <line x1="65" y1="15" x2="65" y2="105" stroke="currentColor" />
        <line x1="65" y1="105" x2="875" y2="105" stroke="currentColor" />
        {indices.map((index) => {
          let path = '';
          let move = true;
          data.positions.forEach((row, sampleIndex) => {
            const value = row[index];
            const timestamp = data.timestamps[sampleIndex];
            if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(timestamp)) {
              move = true;
              return;
            }
            path += `${move ? 'M' : 'L'} ${x(timestamp)} ${y(value)} `;
            move = false;
          });
          return (
            <path key={index} d={path} fill="none" stroke={jointColor(index)} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round">
              <title>{data.joint_names[index]}</title>
            </path>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <text key={fraction} x={65 + fraction * 810} y="125" textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>
            {(duration * fraction).toFixed(1)}
          </text>
        ))}
        <text x="470" y="145" textAnchor="middle">Time (s)</text>
      </svg>
    </div>
  );
}

export default function DatasetJointChart({ dataset, index }: Props) {
  const [data, setData] = useState<EpisodeJoints | null>(null);
  const [jointIndex, setJointIndex] = useState<number | null>(null);
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
              value={jointIndex ?? 'all'}
              onChange={(event) =>
                setJointIndex(event.target.value === 'all' ? null : Number(event.target.value))
              }
            >
              <option value="all">All joints</option>
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
          <div className="datasets-joint-legend" role="group" aria-label="Select joint to inspect">
            {data.joint_names.map((name, value) => (
              <button
                key={name}
                type="button"
                className="datasets-joint-legend-item"
                aria-pressed={jointIndex === value}
                onClick={() => setJointIndex(jointIndex === value ? null : value)}
              >
                <span className="datasets-joint-swatch" style={{ backgroundColor: jointColor(value) }} />
                {name}
              </button>
            ))}
          </div>
          <JointLines data={data} jointIndex={jointIndex} />
          <p className="datasets-note">
            {data.timestamps.length} recorded samples · Stored values;
            units depend on the recording configuration. All joints share one
            vertical scale; select a joint to inspect its own range.
          </p>
        </>
      ) : (
        <p role="status">Loading joint data…</p>
      )}
    </section>
  );
}
