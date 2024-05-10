# Упражнение 4.4

import numpy as np
import matplotlib.pyplot as plt
from thinkdsp import Noise
from thinkdsp import decorate
from pr4_1 import log_log


def normal_prob_plot(sample, **options):
    """Makes a normal probability plot with a fitted line."""

    n = len(sample)
    xs = np.random.normal(0, 1, n)
    xs.sort()

    ys = np.sort(sample)

    mean, std = np.mean(sample), np.std(sample)
    fit_ys = mean + std * xs
    plt.plot(xs, fit_ys, color='gray', alpha=0.5, label='model')

    plt.plot(xs, ys, **options)


class UncorrelatedPoissonNoise(Noise):
    """Uncorrelated Poisson noise."""

    def evaluate(self, ts):
        ys = np.random.poisson(self.amp, len(ts))
        return ys


amp = 0.001
framerate = 10000
duration = 1

signal = UncorrelatedPoissonNoise(amp=amp)
wave = signal.make_wave(duration=duration, framerate=framerate)
wave.make_audio()

expected = amp * framerate * duration
actual = sum(wave.ys)
print(expected, actual)
wave.plot()

spectrum = wave.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)', ylabel='Power', **log_log)
print(spectrum.estimate_slope().slope)

amp = 1
framerate = 10000
duration = 1

signal = UncorrelatedPoissonNoise(amp=amp)
wave = signal.make_wave(duration=duration, framerate=framerate)
wave.make_audio()
wave.plot()

spectrum = wave.make_spectrum()
spectrum.hs[0] = 0

normal_prob_plot(spectrum.real, label='real')
decorate(xlabel='Normal sample', ylabel='Power')

normal_prob_plot(spectrum.imag, label='imag', color='C1')
decorate(xlabel='Normal sample')
