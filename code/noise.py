"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkstats2
import thinkdsp
import thinkplot


def process_noise(signal, root='white'):
    """Plots wave and spectrum for noise signals.

    signal: Signal
    root: string used to generate file names
    """
    framerate = 11025
    wave = signal.make_wave(duration=0.5, framerate=framerate)

    # 0: waveform
    segment = wave.segment(duration=0.1)
    segment.plot(linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise0',
                   xlabel='time (s)',
                   ylabel='amplitude',
                   ylim=[-1.05, 1.05])

    spectrum = wave.make_spectrum()

    # 1: spectrum
    spectrum.plot_power(linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise1',
                   xlabel='frequency (Hz)',
                   ylabel='power density',
                   xlim=[0, framerate/2])

    slope, _, _, _, _ = spectrum.estimate_slope()
    print('estimated slope', slope)

    # 2: integrated spectrum
    integ = spectrum.make_integrated_spectrum()
    integ.plot_power()
    thinkplot.save(root=root+'noise2',
                   xlabel='frequency (Hz)',
                   ylabel='cumulative fraction of total power',
                   xlim=[0, framerate/2])

    # 3: log-log spectral density
    spectrum.plot_power(low=1, linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise3',
                   xlabel='frequency (Hz)',
                   ylabel='power density',
                   xscale='log',
                   yscale='log',
                   xlim=[0, framerate/2])


def plot_power_density(root, spectrum):
    """
    """
    # 4: CDF of power density
    cdf = thinkstats2.MakeCdfFromList(spectrum.power)
    thinkplot.cdf(cdf)
    thinkplot.save(root=root+'noise4',
                   xlabel='power density',
                   ylabel='CDF')

    # 5: CCDF of power density, log-y
    thinkplot.cdf(cdf, complement=True)
    thinkplot.save(root=root+'noise5',
                   xlabel='power density',
                   ylabel='log(CCDF)',
                   yscale='log')


def plot_gaussian_noise():
    """Shows the distribution of the spectrum of Gaussian noise.
    """
    thinkdsp.random_seed(18)
    signal = thinkdsp.UncorrelatedGaussianNoise()
    wave = signal.make_wave(duration=0.5, framerate=11025)
    spectrum = wave.make_spectrum()

    thinkplot.preplot(2, cols=2)
    thinkstats2.NormalProbabilityPlot(spectrum.real, label='real')
    thinkplot.config(xlabel='normal sample',
                     ylabel='power density',
                     ylim=[-250, 250],
                     loc='lower right')

    thinkplot.subplot(2)
    thinkstats2.NormalProbabilityPlot(spectrum.imag, label='imag')
    thinkplot.config(xlabel='normal sample',
                     ylim=[-250, 250],
                     loc='lower right')

    thinkplot.save(root='noise1')


def plot_pink_noise():
    """Makes a plot showing power spectrums for pink noise.
    """
    thinkdsp.random_seed(20)

    duration = 1.0
    framerate = 512

    signal = thinkdsp.UncorrelatedUniformNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    white = wave.make_spectrum()

    signal = thinkdsp.PinkNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    pink = wave.make_spectrum()

    signal = thinkdsp.BrownianNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    red = wave.make_spectrum()

    linewidth = 1
    white.plot_power(low=1, label='white', color='gray', linewidth=linewidth)
    pink.plot_power(low=1, label='pink', color='pink', linewidth=linewidth)
    red.plot_power(low=1, label='red', color='red', linewidth=linewidth)
    thinkplot.save(root='noise-triple',
                   xlabel='frequency (Hz)',
                   ylabel='power',
                   xscale='log',
                   yscale='log',
                   axis=[1, 300, 1e-4, 1e5])


def main():
    plot_gaussian_noise()

    thinkdsp.random_seed(20)
    signal = thinkdsp.UncorrelatedUniformNoise()
    process_noise(signal, root='white')

    thinkdsp.random_seed(20)
    signal = thinkdsp.PinkNoise(beta=1.0)
    process_noise(signal, root='pink')

    thinkdsp.random_seed(17)
    signal = thinkdsp.BrownianNoise()
    process_noise(signal, root='red')

    plot_pink_noise()


if __name__ == '__main__':
    main()
