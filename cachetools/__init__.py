"""Minimal cachetools TTLCache implementation for offline environments."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from typing import Dict, Generic, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Simplified TTL cache compatible with cachetools.TTLCache interface."""

    def __init__(
        self,
        maxsize: int,
        ttl: float,
        *,
        timer: Optional[callable] = None,
        getsizeof: Optional[callable] = None,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if getsizeof is not None:
            raise NotImplementedError("getsizeof is not supported in this stub")

        self._maxsize = maxsize
        self._ttl = ttl
        self._timer = timer or time.monotonic
        self._store: Dict[K, Tuple[V, float]] = {}
        self._order: list[K] = []
        self._lock = threading.RLock()

    def _purge(self) -> None:
        now = self._timer()
        expired = [key for key, (_, exp) in self._store.items() if exp <= now]
        for key in expired:
            self._store.pop(key, None)
            try:
                self._order.remove(key)
            except ValueError:
                pass

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._maxsize:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

    def __getitem__(self, key: K) -> V:
        with self._lock:
            self._purge()
            value, _ = self._store[key]
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return value

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._purge()
            expires = self._timer() + self._ttl
            self._store[key] = (value, expires)
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            self._evict_if_needed()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            self._purge()
            return key in self._store

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            self._purge()
            if key not in self._store:
                if default is not None:
                    return default
                raise KeyError(key)
            self._order.remove(key)
            value, _ = self._store.pop(key)
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge()
            return len(self._store)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            self._purge()
            return iter(list(self._order))

    def items(self) -> Iterable[Tuple[K, V]]:
        with self._lock:
            self._purge()
            return [(key, self._store[key][0]) for key in self._order]


__all__ = ["TTLCache"]
