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

PI2 = np.pi * 2
GRAY = '0.7'


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


def plot_sawtooth_and_spectrum(wave, root):
    """Makes a plot showing a sawtoothwave and its spectrum.
    """
    thinkplot.preplot(cols=2)
    wave.plot()
    thinkplot.config(xlabel='Time (s)')

    thinkplot.subplot(2)
    spectrum = wave.make_spectrum()
    spectrum.plot()
    thinkplot.config(xlabel='Frequency (Hz)',
                     #ylabel='Amplitude',
                     xlim=[0, spectrum.fs[-1]])

    thinkplot.save(root)


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


def plot_filters(close):
    """Plots the filter that corresponds to diff, deriv, and integral.
    """
    thinkplot.preplot(3, cols=2)

    diff_window = np.array([1.0, -1.0])
    diff_filter = make_filter(diff_window, close)
    diff_filter.plot(label='diff')

    deriv_filter = close.make_spectrum()
    deriv_filter.hs = PI2 * 1j * deriv_filter.fs
    deriv_filter.plot(label='derivative')

    thinkplot.config(xlabel='Frequency (1/day)',
                     ylabel='Amplitude ratio',
                     loc='upper left')

    thinkplot.subplot(2)
    integ_filter = deriv_filter.copy()
    integ_filter.hs = 1 / (PI2 * 1j * integ_filter.fs)

    integ_filter.plot(label='integral')
    thinkplot.config(xlabel='Frequency (1/day)',
                     ylabel='Amplitude ratio', 
                     yscale='log')
    thinkplot.save('diff_int3')


def plot_diff_deriv(close):
    change = thinkdsp.Wave(np.diff(close.ys), framerate=1)

    deriv_spectrum = close.make_spectrum().differentiate()
    deriv = deriv_spectrum.make_wave()

    low, high = 0, 50
    thinkplot.preplot(2)
    thinkplot.plot(change.ys[low:high], label='diff')
    thinkplot.plot(deriv.ys[low:high], label='derivative')

    thinkplot.config(xlabel='Time (day)', ylabel='Price change ($)')
    thinkplot.save('diff_int4')

    
def plot_integral(close):

    deriv_spectrum = close.make_spectrum().differentiate()

    integ_spectrum = deriv_spectrum.integrate()
    print(integ_spectrum.hs[0])
    integ_spectrum.hs[0] = 0
    
    thinkplot.preplot(2)
    integ_wave = integ_spectrum.make_wave()
    close.plot(label='closing prices')
    integ_wave.plot(label='integrated derivative')
    thinkplot.config(xlabel='Time (day)', ylabel='Price ($)', 
                     legend=True, loc='upper left')

    thinkplot.save('diff_int5')

    
def plot_ratios(in_wave, out_wave):

    # compare filters for cumsum and integration
    diff_window = np.array([1.0, -1.0])
    padded = thinkdsp.zero_pad(diff_window, len(in_wave))
    diff_wave = thinkdsp.Wave(padded, framerate=in_wave.framerate)
    diff_filter = diff_wave.make_spectrum()
    
    cumsum_filter = diff_filter.copy()
    cumsum_filter.hs = 1 / cumsum_filter.hs
    cumsum_filter.plot(label='cumsum filter', color=GRAY, linewidth=7)
    
    integ_filter = cumsum_filter.copy()
    integ_filter.hs = integ_filter.framerate / (PI2 * 1j * integ_filter.fs)
    integ_filter.plot(label='integral filter')

    thinkplot.config(xlim=[0, integ_filter.max_freq],
                     yscale='log', legend=True)
    thinkplot.save('diff_int8')

    # compare cumsum filter to actual ratios
    cumsum_filter.plot(label='cumsum filter', color=GRAY, linewidth=7)
    
    in_spectrum = in_wave.make_spectrum()
    out_spectrum = out_wave.make_spectrum()
    ratio_spectrum = out_spectrum.ratio(in_spectrum, thresh=1)
    ratio_spectrum.plot(label='ratio', style='.', markersize=4)

    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, integ_filter.max_freq],
                     yscale='log', legend=True)
    thinkplot.save('diff_int9')



def plot_diff_filters(wave):

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
    plot_wave_and_spectrum(close, root='diff_int1')

    change = thinkdsp.Wave(np.diff(ys), framerate=1)
    plot_wave_and_spectrum(change, root='diff_int2')

    plot_filters(close)

    plot_diff_deriv(close)

    signal = thinkdsp.SawtoothSignal(freq=50)
    in_wave = signal.make_wave(duration=0.1, framerate=44100)
    plot_sawtooth_and_spectrum(in_wave, 'diff_int6')

    out_wave = in_wave.cumsum()
    out_wave.unbias()
    plot_sawtooth_and_spectrum(out_wave, 'diff_int7')

    plot_integral(close)
    plot_ratios(in_wave, out_wave)


if __name__ == '__main__':
    main()
