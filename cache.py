"""
Tiny thread-safe in-memory TTL cache.

FIRMS updates every few hours and OpenAQ's India sweep pages through up to
25 requests, so re-fetching on every page load is slow and burns API quota.
Results are cached per-process for a short time instead.
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

    def get_or_set(self, key: Hashable, producer: Callable[[], Any]) -> Any:
        """Return the cached value for key, calling producer() if missing/stale.

        Exceptions from producer() propagate and nothing is cached, so a
        transient API failure is retried on the next request.
        """
        now = self._clock()
        with self._lock:
            hit = self._store.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1]

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
