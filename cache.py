"""
Tiny thread-safe in-memory TTL cache.

FIRMS updates every few hours, OpenAQ's India sweep pages through up to 25
requests, and the wind attribution takes a few seconds of CPU, so results are
cached per-process for a short time. Concurrent requests for the same key
wait for a single computation instead of all hitting the upstream APIs.
"""

import threading
import time
from typing import Any, Callable, Dict, Hashable, Tuple


class TTLCache:
    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self._clock = clock
        self._store: Dict[Hashable, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._key_locks: Dict[Hashable, threading.Lock] = {}

    def _fresh(self, key):
        hit = self._store.get(key)
        if hit and self._clock() - hit[0] < self.ttl:
            return True, hit[1]
        return False, None

    def get_or_set(self, key: Hashable, producer: Callable[[], Any]) -> Any:
        """Return the cached value for key, calling producer() if missing/stale.

        Exceptions from producer() propagate and nothing is cached, so a
        transient API failure is retried on the next request.
        """
        with self._lock:
            ok, value = self._fresh(key)
            if ok:
                return value
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        with key_lock:
            with self._lock:
                ok, value = self._fresh(key)   # another thread may have filled it
                if ok:
                    return value
            value = producer()
            with self._lock:
                self._store[key] = (self._clock(), value)
            return value

    def fetched_at(self, key: Hashable):
        """Monotonic timestamp the key was stored at, or None."""
        with self._lock:
            hit = self._store.get(key)
            return hit[0] if hit else None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
