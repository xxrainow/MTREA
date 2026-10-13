import { hardwareRequest } from './hardware';

export type CameraConfig = {
  name: string;
  width: number;
  height: number;
  fps: number;
};

export type CameraStatus = {
  running: boolean;
  cameras: string[];
  error: string | null;
};

export function cameraRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  return hardwareRequest<T>(`/api/cameras/${path}`, options);
}
