"""Causal, stateful EMG filters. Requires scipy; no resampling or unit conversion."""
import numpy as np


class EMGFilter:
    def __init__(self, sample_rate, high=500, notch=True):
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError('EMG filtering requires a positive, regular sampling rate.')
        if not 20 < high < sample_rate / 2:
            raise ValueError(f'20–{high:g} Hz is unavailable at {sample_rate:g} samples/s. '
                             f'The upper cutoff must be below {sample_rate/2:g} Hz.')
        if notch and sample_rate <= 120:
            raise ValueError('A 60 Hz notch requires a sample rate above 120 samples/s.')
        try:
            from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi
        except ImportError as exc:
            raise RuntimeError('Install filters with: python -m pip install scipy') from exc
        self.sos = butter(4, [20, high], btype='bandpass', fs=sample_rate, output='sos')
        if notch:
            b, a = iirnotch(60, Q=30, fs=sample_rate)
            self.sos = np.vstack([self.sos, tf2sos(b, a)])
        self.filter = sosfilt
        self.initial = sosfilt_zi(self.sos)
        self.sample_rate = sample_rate
        self.state = None
        self.last_time = None
        self.resets = 0

    def process(self, times, samples):
        times = np.asarray(times, dtype=float)
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or len(times) != len(samples):
            raise ValueError('Expected one timestamp per row of channel samples.')
        output = np.full_like(samples, np.nan)
        if not len(times):
            return output
        valid = np.isfinite(times) & np.all(np.isfinite(samples), axis=1)
        previous = np.r_[self.last_time if self.last_time is not None else times[0], times[:-1]]
        delta = times - previous
        breaks = (delta <= 0) | (delta > 1.5 / self.sample_rate) | ~valid
        # Never carry filter state across an invalid sample or missing interval.
        starts = np.flatnonzero(breaks | np.r_[True, ~valid[:-1]])
        ends = np.r_[starts[1:], len(times)]
        for start, end in zip(starts, ends):
            if not valid[start]:
                self.state = None
                continue
            if breaks[start] or self.state is None:
                if self.last_time is not None or start > 0:
                    self.resets += 1
                self.state = self.initial[:, :, None] * samples[start][None, None, :]
            output[start:end], self.state = self.filter(self.sos, samples[start:end], axis=0, zi=self.state)
        self.last_time = float(times[-1]) if valid[-1] else None
        return output
