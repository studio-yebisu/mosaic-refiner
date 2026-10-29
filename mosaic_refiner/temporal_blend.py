"""Bounded crossfade of overlapping temporal predictions, independent of Torch."""


def temporal_tiles(count, chunk_frames, halo_frames):
    if count < 1 or chunk_frames < 1 or not 0 <= halo_frames < chunk_frames / 2:
        raise ValueError("temporal tile dimensions are invalid")
    import numpy as np
    for first in range(0, count, chunk_frames):
        end = min(count, first + chunk_frames)
        start, stop = max(0, first-halo_frames), min(count, end+halo_frames)
        indices = np.arange(start, stop)
        weights = np.ones(len(indices), np.float32)
        if halo_frames:
            if start > 0:
                weights = np.minimum(weights, (indices-start+.5)/(2*halo_frames))
            if end < count:
                weights = np.minimum(weights, (end+halo_frames-indices-.5)/(2*halo_frames))
        # A later tile can touch frames >= end-halo, so those stay buffered.
        flush_end = max(start, end-halo_frames) if end < count else count
        yield first, end, start, stop, weights.astype(np.float32), flush_end


class TemporalBlend:
    def __init__(self):
        self.pending = {}
        self.next_frame = 0
        self.max_buffered_frames = 0

    def add(self, start, frames, weights):
        import numpy as np
        for index, (frame, weight) in enumerate(zip(frames, weights), start):
            if index < self.next_frame:
                continue
            prediction = np.asarray(frame, np.float32) * weight
            if index in self.pending:
                value, total = self.pending[index]
                self.pending[index] = (value + prediction, total + weight)
            else:
                self.pending[index] = (prediction, weight)
        self.max_buffered_frames = max(self.max_buffered_frames, len(self.pending))

    def flush(self, end):
        import numpy as np
        while self.next_frame < end:
            index = self.next_frame
            value, weight = self.pending.pop(index)
            if weight <= 0:
                raise ValueError("temporal blend has zero support")
            self.next_frame += 1
            yield index, np.rint(value / weight).clip(0,255).astype(np.uint8)
