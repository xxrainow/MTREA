import { useState } from 'react';
import { cameraRequest } from '../api/cameras';
import type { CameraConfig, CameraStatus } from '../api/cameras';
import { useHardwareAction, useHardwarePoll } from './useHardware';

async function loadCameras(signal: AbortSignal) {
  const [configs, status] = await Promise.all([
    cameraRequest<CameraConfig[]>('config', { signal }),
    cameraRequest<CameraStatus>('status', { signal }),
  ]);

  return { configs, status };
}

export function useCameras() {
  const { data, error: pollError, refresh } = useHardwarePoll(loadCameras);
  const { act, pending, error: actionError } = useHardwareAction(refresh);
  const [streamVersion, setStreamVersion] = useState(0);

  const running = data?.status.running ?? false;
  const unavailable = !data || Boolean(pollError);

  function togglePreview() {
    const action = running ? 'stop' : 'start';

    void act(
      running ? 'Stopping preview' : 'Starting preview',
      async () => {
        await cameraRequest<CameraStatus>(action, { method: 'POST' });

        if (action === 'start') {
          setStreamVersion((version) => version + 1);
        }
      },
    );
  }

  return {
    data,
    pollError,
    actionError,
    pending,
    running,
    unavailable,
    streamVersion,
    togglePreview,
  };
}

export type CameraControls = ReturnType<typeof useCameras>;
