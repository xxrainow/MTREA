export type TrainingConfig = {
  target: 'ssh' | 'local'; dataset: string; seed: string; steps: string;
  batchSize: string; learningRate: string; runName: string;
  warmupSteps: string; saveFreq: string; logFreq: string; wandb: boolean;
};
export const defaultConfig: TrainingConfig = {
  target: 'ssh', dataset: '', seed: '', steps: '10000', batchSize: '16',
  learningRate: '0.000025', runName: 'h0_pickplace_seed0', warmupSteps: '1000',
  saveFreq: '1000', logFreq: '200', wandb: false,
};
export type TrainingMetric = { step: number; loss: number; lr: number };
export type TrainingRun = {
  id: string; config: TrainingConfig; status: 'running' | 'disconnected' | 'completed' | 'stopped'; step: number; metrics: TrainingMetric[]; logs: string[];
};
export type Checkpoint = { id: string; run: TrainingRun; step: number; final: boolean };
export type TrainingState = {
  config: TrainingConfig; checked: boolean; errors: string[];
  run: TrainingRun | null; checkpoints: Checkpoint[]; runs: TrainingRun[];
};
export const initialTrainingState: TrainingState = {
  config: defaultConfig, checked: false, errors: [], run: null, checkpoints: [], runs: [],
};
export function validateConfig(config: TrainingConfig, checkpoints: Checkpoint[]): string[] {
  const errors: string[] = [];
  if (config.dataset !== 'demo/pick_place') errors.push('Select the simulated T_sub dataset.');
  if (!config.seed.trim() || !Number.isSafeInteger(Number(config.seed)) || Number(config.seed) < 0) errors.push('Seed must be an explicit non-negative integer.');
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(config.runName)) errors.push('Run name may contain letters, numbers, underscores and hyphens.');
  for (const field of ['steps', 'batchSize', 'warmupSteps', 'saveFreq', 'logFreq'] as const) {
    if (!Number.isSafeInteger(Number(config[field])) || Number(config[field]) <= 0) errors.push(`${field} must be a positive integer.`);
  }
  if (Number(config.warmupSteps) >= Number(config.steps)) errors.push('Warmup steps must be smaller than training steps.');
  if (!Number.isFinite(Number(config.learningRate)) || Number(config.learningRate) <= 0) errors.push('Learning rate must be positive and finite.');
  if (config.target === 'ssh' && Number(config.batchSize) > 16) errors.push('The A100 MIG recipe supports batch sizes up to 16.');
  if (checkpoints.some(c => c.run.config.runName === config.runName)) errors.push('This demo run name already has saved results. Choose a new name.');
  if (Number(config.steps) / Number(config.saveFreq) > 2000) errors.push('The demo supports at most 2,000 save intervals per run.');
  return errors;
}
export function isTrainingActive(run: TrainingRun | null): boolean {
  return run?.status === 'running' || run?.status === 'disconnected';
}
export type TrainingAction =
  | { type: 'configure'; config: TrainingConfig }
  | { type: 'check' }
  | { type: 'start'; id: string }
  | { type: 'tick' | 'advance' | 'complete' | 'disconnect' | 'reconnect' | 'stop' | 'new' };

export function trainingReducer(state: TrainingState, action: TrainingAction): TrainingState {
  if (action.type === 'configure') return isTrainingActive(state.run) ? state : { ...state, config: action.config, checked: false, errors: [] };
  if (action.type === 'check') {
    const errors = validateConfig(state.config, state.checkpoints);
    return { ...state, errors, checked: errors.length === 0 };
  }
  if (action.type === 'start') {
    if (!state.checked || isTrainingActive(state.run) || validateConfig(state.config, state.checkpoints).length) return state;
    const run: TrainingRun = { id: action.id, config: { ...state.config }, status: 'running', step: 0, metrics: [], logs: ['SIMULATION started · No training process launched'] };
    return { ...state, checked: false, run, runs: [...state.runs, run] };
  }
  if (action.type === 'new') return isTrainingActive(state.run) ? state : { ...state, run: null, checked: false };
  const run = state.run;
  if (!run || !isTrainingActive(run)) return state;
  if (action.type === 'disconnect') return updateRun(state, { ...run, status: 'disconnected', logs: [...run.logs, 'SIMULATION connection lost'].slice(-200) });
  if (action.type === 'reconnect') return updateRun(state, { ...run, status: 'running', logs: [...run.logs, 'SIMULATION reconnected'].slice(-200) });
  // A lost connection gives no evidence of completion or successful cancellation.
  if (run.status === 'disconnected') return state;
  if (action.type === 'stop') return updateRun(state, { ...run, status: 'stopped', logs: [...run.logs, 'SIMULATION stopped'].slice(-200) });
  const total = Number(run.config.steps);
  const step = action.type === 'complete' ? total : Math.min(action.type === 'tick' ? total : total - 1, run.step + Math.max(1, Math.floor(total / (action.type === 'tick' ? 100 : 5))));
  const completed = step === total;
  const warmup = Number(run.config.warmupSteps);
  const peak = Number(run.config.learningRate);
  const lr = step < warmup ? peak * step / warmup : peak / 10 + peak * .9 * (1 + Math.cos(Math.PI * (step - warmup) / (total - warmup))) / 2;
  const loss = .12 + .9 * Math.exp(-5 * step / total) + .015 * Math.sin(step / total * 80);
  const metrics = [...run.metrics, { step, loss, lr }].slice(-500);
  const logs = [...run.logs, `SIMULATION step=${step}/${total} loss=${loss.toFixed(4)} lr=${lr.toExponential(3)}`, ...(completed ? ['SIMULATION completed · Final snapshot saved'] : [])].slice(-200);
  const updated: TrainingRun = { ...run, step, metrics, logs, status: completed ? 'completed' : 'running' };
  const checkpoints = [...state.checkpoints];
  const interval = Number(run.config.saveFreq);
  const savedSteps: number[] = [];
  for (let saved = (Math.floor(run.step / interval) + 1) * interval; saved <= step; saved += interval) savedSteps.push(saved);
  if (completed && !savedSteps.includes(total)) savedSteps.push(total);
  for (const savedStep of savedSteps) {
    const id = `${run.id}:${savedStep}`;
    if (!checkpoints.some(c => c.id === id)) checkpoints.push({ id, run: updated, step: savedStep, final: completed && savedStep === total });
  }
  return { ...updateRun(state, updated), checkpoints };
}

function updateRun(state: TrainingState, run: TrainingRun): TrainingState {
  return { ...state, run, runs: state.runs.map(item => item.id === run.id ? run : item) };
}
