"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd

import thinkdsp
import thinkplot


def plot_wave_and_spectrum(wave, root):
    """Makes a plot showing a wave and its spectrum.

    wave: Wave object
    root: string used to generate filenames
    """
    thinkplot.preplot(cols=2)
    wave.plot()
    thinkplot.config(xlabel='Time (days)',
                     ylabel='Price ($)')

    thinkplot.subplot(2)
    spectrum = wave.make_spectrum()
    print(spectrum.estimate_slope())
    spectrum.plot()
    thinkplot.config(xlabel='Frequency (1/days)',
                     ylabel='Amplitude',
                     xlim=[0, spectrum.fs[-1]],
                     xscale='log',
                     yscale='log')

    thinkplot.save(root=root)


def make_filter(window, wave):
    """Computes the filter that corresponds to a window.
    
    window: NumPy array
    wave: wave used to choose the length and framerate
    
    returns: new Spectrum
    """
    padded = thinkdsp.zero_pad(window, len(wave))
    window_wave = thinkdsp.Wave(padded, framerate=wave.framerate)
    window_spectrum = window_wave.make_spectrum()
    return window_spectrum


def plot_diff_filter(close):
    """Plots the filter that corresponds to first order finite difference.
    """
    diff_window = np.array([1.0, -1.0])
    diff_filter = make_filter(diff_window, close)
    diff_filter.plot()
    thinkplot.config(xlabel='Frequency (1/day)', ylabel='Amplitude ratio')
    thinkplot.save('diff_int3')


def plot_ratios(wave, wave2):
    spectrum = wave.make_spectrum()
    spectrum2 = wave2.make_spectrum()
    
    amps = spectrum.amps
    amps2 = spectrum2.amps

    n = min(len(amps), len(amps2))
    ratio = amps2[:n] / amps[:n]

    thinkplot.preplot(1)
    thinkplot.plot(ratio, label='ratio')

    window = np.array([1.0, -1.0])
    padded = thinkdsp.zero_pad(window, len(wave))
    fft_window = np.fft.rfft(padded)
    thinkplot.plot(abs(fft_window), color='0.7', label='filter')

    thinkplot.config(xlabel='Frequency (1/days)',
                     #xlim=[0, spectrum.fs[-1]],
                     ylabel='Amplitude ratio',
                     #ylim=[0, 4],
                     loc='upper left')
    thinkplot.save(root='diff_int3')


def plot_derivative(wave, wave2):
    # compute the derivative by spectral decomposition
    spectrum = wave.make_spectrum()
    spectrum3 = wave.make_spectrum()
    spectrum3.differentiate()
    
    # plot the derivative computed by diff and differentiate
    wave3 = spectrum3.make_wave()
    wave2.plot(color='0.7', label='diff')
    wave3.plot(label='derivative')
    thinkplot.config(xlabel='days',
                     xlim=[0, 1650],
                     ylabel='dollars',
                     loc='upper left')

    thinkplot.save(root='diff_int4')

    # plot the amplitude ratio compared to the diff filter
    amps = spectrum.amps
    amps3 = spectrum3.amps
    ratio3 = amps3 / amps

    thinkplot.preplot(1)
    thinkplot.plot(ratio3, label='ratio')

    window = np.array([1.0, -1.0])
    padded = thinkdsp.zero_pad(window, len(wave))
    fft_window = np.fft.rfft(padded)
    thinkplot.plot(abs(fft_window), color='0.7', label='filter')

    thinkplot.config(xlabel='frequency (1/days)',
                     xlim=[0, 1650/2],
                     ylabel='amplitude ratio',
                     #ylim=[0, 4],
                     loc='upper left')
    thinkplot.save(root='diff_int5')


def plot_filters(wave):

    window1 = np.array([1, -1])
    window2 = np.array([-1, 4, -3]) / 2.0
    window3 = np.array([2, -9, 18, -11]) / 6.0
    window4 = np.array([-3, 16, -36, 48, -25]) / 12.0
    window5 = np.array([12, -75, 200, -300, 300, -137]) / 60.0

    thinkplot.preplot(5)
    for i, window in enumerate([window1, window2, window3, window4, window5]):
        padded = thinkdsp.zero_pad(window, len(wave))
        fft_window = np.fft.rfft(padded)
        n = len(fft_window)
        thinkplot.plot(abs(fft_window)[:], label=i+1)

    thinkplot.show()




def main():
    names = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv('fb.csv', header=0, names=names, parse_dates=[0])
    ys = df.close.values[::-1]
    close = thinkdsp.Wave(ys, framerate=1)
    #plot_wave_and_spectrum(close, root='diff_int1')

    change = thinkdsp.Wave(np.diff(ys), framerate=1)
    #plot_wave_and_spectrum(change, root='diff_int2')

    plot_diff_filter(close)
    return

    plot_derivative(wave, wave2)

    plot_filters(wave)


if __name__ == '__main__':
    main()
