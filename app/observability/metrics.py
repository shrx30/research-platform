import time
from contextlib import contextmanager
from threading import Lock


class MetricsCollector:
    def __init__(self):
        self._lock = Lock()
        self._latencies: dict[str, float] = {}

    def reset(self) -> None:
        with self._lock:
            self._latencies = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start

            with self._lock:
                self._latencies[name] = round(duration, 3)

            print(
                f"[LATENCY] {name}: {duration:.3f}s"
            )

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latencies)


metrics = MetricsCollector()