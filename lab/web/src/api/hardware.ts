export type Arm = 'follower' | 'leader';
export type Backend = 'real' | 'fake';

export type ArmConfig = {
  error?: string;
  port?: string;
  port_present?: boolean | null;
  id?: string;
  use_degrees?: boolean;
  calibration_present?: boolean;
  legacy_calibration_present?: boolean;
  calibration_file?: string;
};

export type HardwareConfig = {
  backend: Backend;
  arms: Record<Arm, ArmConfig>;
  ports: string[] | null;
  joint_order: string[];
  lock: { owner: string } | null;
};

export type CalibrationStatus = {
  backend: Backend;
  state: string;
  arm: Arm | null;
  active: boolean;
  can_next: boolean;
  instruction: string;
  error: string | null;
  saved_path: string | null;
  joints: Record<string, { current: number; min: number | null; max: number | null }>;
};

export type TeleopStatus = {
  backend: Backend;
  running: boolean;
  fps: number;
  target_fps: number;
  error: string | null;
  torque_on_stop: string;
};

export type JointFrame =
  | { type: 'teleop'; leader: Record<string, number>; follower: Record<string, number> }
  | { type: 'calibration'; arm: Arm; state: string; joints: CalibrationStatus['joints'] };

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Hardware request failed.';
}

export async function hardwareRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed: HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function hardwarePost<T>(path: string, body?: unknown): Promise<T> {
  return hardwareRequest<T>(path, {
    method: 'POST',
    ...(body === undefined ? {} : {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  });
}

export async function loadDevices(signal: AbortSignal) {
  const [config, calibration] = await Promise.all([
    hardwareRequest<HardwareConfig>('/api/hardware/config', { signal }),
    hardwareRequest<CalibrationStatus>('/api/calibration/status', { signal }),
  ]);
  return { config, calibration };
}

export async function loadTeleop(signal: AbortSignal) {
  const [config, teleop] = await Promise.all([
    hardwareRequest<HardwareConfig>('/api/hardware/config', { signal }),
    hardwareRequest<TeleopStatus>('/api/teleop/status', { signal }),
  ]);
  return { config, teleop };
}

export function jointLabel(joint: string): string {
  return joint.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase());
}
