"""Bounded producer/consumer overlap with explicit cancellation and ownership."""
from queue import Queue, Empty, Full
from threading import Event, Thread
import time


def interval_overlap_seconds(left, right):
    """Linear sweep after sorting, for two internally disjoint stage streams."""
    left, right = sorted(left), sorted(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


class BoundedStream:
    def __init__(self, factory, capacity=1, name="mosaic-producer", cancelled=lambda: False):
        if capacity < 1:
            raise ValueError("queue capacity must be positive")
        self.factory, self.cancelled = factory, cancelled
        self.queue = Queue(capacity)
        self.stop = Event()
        self.thread = Thread(target=self._produce, name=name, daemon=False)
        self.stats = {"capacity": capacity, "produced": 0, "consumed": 0,
                      "max_queued": 0, "producer_seconds": 0.0, "consumer_wait_seconds": 0.0}

    def _put(self, item):
        while not self.stop.is_set() and not self.cancelled():
            try:
                self.queue.put(item, timeout=.05)
                self.stats["max_queued"] = max(self.stats["max_queued"], self.queue.qsize())
                return True
            except Full:
                pass
        return False

    def _produce(self):
        started = time.monotonic()
        iterator = None
        try:
            iterator = iter(self.factory())
            for item in iterator:
                if not self._put(("value", item)):
                    break
                self.stats["produced"] += 1
            self._put(("end", None))
        except BaseException as exc:
            self._put(("error", exc))
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
            self.stats["producer_seconds"] = time.monotonic() - started

    def __enter__(self):
        self.thread.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        started = time.monotonic()
        while not self.stop.is_set():
            if self.cancelled():
                raise KeyboardInterrupt
            try:
                kind, value = self.queue.get(timeout=.05)
            except Empty:
                if not self.thread.is_alive():
                    # The producer can enqueue its final values immediately
                    # after the timed get expires. Drain before declaring EOF.
                    try:kind,value=self.queue.get_nowait()
                    except Empty:raise StopIteration
                else:continue
            self.stats["consumer_wait_seconds"] += time.monotonic() - started
            if kind == "end":raise StopIteration
            if kind == "error":raise value
            self.stats["consumed"] += 1
            return value
        raise StopIteration

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()

    def __exit__(self, *_):
        self.close()
