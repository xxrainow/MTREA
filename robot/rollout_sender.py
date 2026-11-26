"""One transport call in flight and one replaceable observation waiting to send."""

import time
from copy import deepcopy
from threading import Condition


class LatestObservationSender:
    def __init__(self, send, *, max_age_s, clock=time.monotonic, on_queue_wait=None):
        self._send = send
        self._max_age_s = max_age_s
        self._clock = clock
        self._on_queue_wait = on_queue_wait
        self._condition = Condition()
        self._pending = None
        self._closed = False
        self._in_flight = False
        self._force_next = False
        self._submitted = self._sent = self._replaced = self._stale = 0
        self._generation = self._superseded = 0

    def generation(self):
        with self._condition:
            return self._generation

    def invalidate_pending(self):
        with self._condition:
            self._generation += 1
            self._superseded += int(self._pending is not None)
            self._pending = None
            self._force_next = False

    def submit(self, observation, *, generation=None):
        # Camera implementations may reuse their buffers after this call returns.
        captured = self._clock()
        owned = deepcopy(observation)
        with self._condition:
            if self._closed:
                return False
            if generation is not None and generation != self._generation:
                self._superseded += 1
                return False
            force = self._force_next or getattr(owned, "must_go", False)
            if self._pending is not None:
                force = force or getattr(self._pending[0], "must_go", False)
                self._replaced += 1
            if force:
                owned.must_go = True
            self._force_next = False
            self._pending = (owned, captured)
            self._submitted += 1
            self._condition.notify()
        return True

    def run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        return
                    observation, captured = self._pending
                    self._pending = None
                    if self._clock() - captured > self._max_age_s:
                        self._stale += 1
                        self._force_next |= getattr(observation, "must_go", False)
                        continue
                    self._in_flight = True
                try:
                    if self._on_queue_wait is not None:
                        self._on_queue_wait(self._clock() - captured)
                    sent = self._send(observation)
                    with self._condition:
                        self._sent += int(bool(sent))
                finally:
                    with self._condition:
                        self._in_flight = False
        finally:
            self.close()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._force_next = False
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {"pending": int(self._pending is not None),
                    "in_flight": self._in_flight, "closed": self._closed,
                    "submitted": self._submitted, "sent": self._sent,
                    "replaced": self._replaced, "stale": self._stale,
                    "superseded": self._superseded}
