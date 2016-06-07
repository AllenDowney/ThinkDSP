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
                   xlabel='Time (s)',
                   ylim=[-1.05, 1.05])

    spectrum = wave.make_spectrum()

    # 1: spectrum
    spectrum.plot_power(linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise1',
                   xlabel='Frequency (Hz)',
                   ylabel='Power',
                   xlim=[0, spectrum.fs[-1]])

    slope, _, _, _, _ = spectrum.estimate_slope()
    print('estimated slope', slope)

    # 2: integrated spectrum
    integ = spectrum.make_integrated_spectrum()
    integ.plot_power()
    thinkplot.save(root=root+'noise2',
                   xlabel='Frequency (Hz)',
                   ylabel='Cumulative fraction of total power',
                   xlim=[0, framerate/2])

    # 3: log-log power spectrum
    spectrum.hs[0] = 0
    thinkplot.preplot(cols=2)
    spectrum.plot_power(linewidth=1, alpha=0.5)
    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Power',
                     xlim=[0, framerate/2])

    thinkplot.subplot(2)
    spectrum.plot_power(linewidth=1, alpha=0.5)
    thinkplot.config(xlabel='Frequency (Hz)',                  
                   xscale='log',
                   yscale='log',
                   xlim=[0, framerate/2])

    thinkplot.save(root=root+'noise3')


def plot_gaussian_noise():
    """Shows the distribution of the spectrum of Gaussian noise.
    """
    thinkdsp.random_seed(18)
    signal = thinkdsp.UncorrelatedGaussianNoise()
    wave = signal.make_wave(duration=0.5, framerate=11025)
    spectrum = wave.make_spectrum()

    thinkplot.preplot(2, cols=2)
    thinkstats2.NormalProbabilityPlot(spectrum.real, label='real')
    thinkplot.config(xlabel='Normal sample',
                     ylabel='Amplitude',
                     ylim=[-250, 250],
                     loc='lower right')

    thinkplot.subplot(2)
    thinkstats2.NormalProbabilityPlot(spectrum.imag, label='imag')
    thinkplot.config(xlabel='Normal sample',
                     ylim=[-250, 250],
                     loc='lower right')

    thinkplot.save(root='noise1')


def plot_pink_noise():
    """Makes a plot showing power spectrums for pink noise.
    """
    thinkdsp.random_seed(20)

    duration = 1.0
    framerate = 512

    def make_spectrum(signal):
        wave = signal.make_wave(duration=duration, framerate=framerate)
        spectrum = wave.make_spectrum()
        spectrum.hs[0] = 0
        return spectrum

    signal = thinkdsp.UncorrelatedUniformNoise()
    white = make_spectrum(signal)

    signal = thinkdsp.PinkNoise()
    pink = make_spectrum(signal)

    signal = thinkdsp.BrownianNoise()
    red = make_spectrum(signal)

    linewidth = 2
    # colorbrewer2.org 4-class sequential OrRd
    white.plot_power(label='white', color='#fdcc8a', linewidth=linewidth)
    pink.plot_power(label='pink', color='#fc8d59', linewidth=linewidth)
    red.plot_power(label='red', color='#d7301f', linewidth=linewidth)
    thinkplot.save(root='noise-triple',
                   xlabel='Frequency (Hz)',
                   ylabel='Power',
                   xscale='log',
                   yscale='log',
                   xlim=[1, red.fs[-1]])


def main():
    thinkdsp.random_seed(17)
    plot_pink_noise()

    thinkdsp.random_seed(17)
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


if __name__ == '__main__':
    main()
