# Write a class called UncorrelatedPoissonNoise that inherits from thinkdsp._Noise and
# provides evaluate. It should use np.random.poisson to generate random values from a
# Poisson distribution. The parameter of this function, lam, is the average number of
# particles during each interval. You can use the attribute amp to specify lam.
# For example, if the frame rate is 10 kHz and amp is 0.001, we expect about 10
# clicks” per second.

# Generate about a second of UP noise and listen to it. For low values of amp,
# like 0.001, it should sound like a Geiger counter. For higher values it should sound
# like white noise. Compute and plot the power spectrum to see whether it looks
# like white noise.

import matplotlib.pyplot as plt
import numpy as np

import code.thinkplot as thinkplot
from code.thinkdsp import Noise

from exercises.lib.lib import play_wave

FRAMERATE = 22050


class UncorrelatedPoissonNoise(Noise):
    """Represents uncorrelated Poisson noise."""

    def __init__(self, amp=None, framerate=None):
        super(UncorrelatedPoissonNoise, self).__init__(amp=amp)
        self.framerate = framerate

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        if self.amp > 1.0:
            raise ValueError('Amp must be 0 > amp <= 1.0')
        ys = np.random.poisson(lam=self.amp, size=len(ts))
        return ys


def run():
    duration = 1.5

    amp = (FRAMERATE / 1000) / FRAMERATE
    print(f'Generating UncorrelatedPoissonNoise w/ low amp = {amp}. Expect "geiger counter".')
    noise_signal = UncorrelatedPoissonNoise(amp=amp, framerate=FRAMERATE)
    wave = noise_signal.make_wave(start=0, duration=duration, framerate=FRAMERATE)
    play_wave(wave)
    print(f'Plotting Spectrum for UncorrelatedPoissonNoise w/ low amp = {amp}')
    spectrum = wave.make_spectrum()
    # TODO Make my own make_spectrum wrapper to fix this
    # NOTE: Fix bug where first element of spectrum is very high and blows up chart scale
    spectrum.hs[0] = 0.
    spectrum.plot_power()
    # TODO Write my own wrapper for log-scale plot
    thinkplot.config(xscale='log', yscale='log')

    amp = 1.0
    print(f'Generating UncorrelatedPoissonNoise w/ high amp = {amp}. Expect "white noise".')
    noise_signal = UncorrelatedPoissonNoise(amp=amp, framerate=FRAMERATE)
    wave = noise_signal.make_wave(start=0, duration=duration, framerate=FRAMERATE)
    play_wave(wave)
    print(f'Plotting Spectrum for UncorrelatedPoissonNoise w/ high amp = {amp}')
    spectrum = wave.make_spectrum()
    # TODO Make my own make_spectrum wrapper to fix this
    spectrum.hs[0] = 0.
    spectrum.plot_power()
    thinkplot.config(xscale='log', yscale='log')


if __name__ == '__main__':
    print("\nChapter 4: ex_4_geiger_counter.py")
    print("****************************")
    run()
