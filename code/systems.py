"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import pandas

import thinkstats2
import thinkdsp
import thinkplot


def plot_wave_and_spectrum(wave, root):
    """Makes a plot showing 
    """
    thinkplot.preplot(cols=2)
    wave.plot()
    thinkplot.config(xlabel='days',
                     xlim=[0, 1650],
                     ylabel='dollars', 
                     legend=False)

    thinkplot.subplot(2)
    spectrum = wave.make_spectrum()
    print(spectrum.estimate_slope())
    spectrum.plot()
    thinkplot.config(xlabel='frequency (1/days)',
                     ylabel='power',
                     xscale='log',
                     yscale='log', 
                     legend=False)

    thinkplot.save(root=root)


def zero_pad(array, n):
    """Makes a new array with the same elements and the given length.

    array: numpy array
    n: length of result

    returns: new NumPy array
    """
    res = numpy.zeros(n)
    res[:len(array)] = array
    return res


def plot_ratios(wave, wave2):
    spectrum = wave.make_spectrum()
    spectrum2 = wave2.make_spectrum()
    
    amps = spectrum.amps
    amps2 = spectrum2.amps

    n = min(len(amps), len(amps2))
    ratio = amps2[:n] / amps[:n]

    thinkplot.preplot(1)
    thinkplot.plot(ratio, label='ratio')

    window = numpy.array([1.0, -1.0])
    padded = zero_pad(window, len(wave))
    fft_window = numpy.fft.rfft(padded)
    thinkplot.plot(abs(fft_window), color='0.7', label='filter')

    thinkplot.config(xlabel='frequency (1/days)',
                     xlim=[0, 1650/2],
                     ylabel='amplitude ratio',
                     ylim=[0, 4],
                     loc='upper left')
    thinkplot.save(root='systems3')


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

    thinkplot.save(root='systems4')

    # plot the amplitude ratio compared to the diff filter
    amps = spectrum.amps
    amps3 = spectrum3.amps
    ratio3 = amps3 / amps

    thinkplot.preplot(1)
    thinkplot.plot(ratio3, label='ratio')

    window = numpy.array([1.0, -1.0])
    padded = zero_pad(window, len(wave))
    fft_window = numpy.fft.rfft(padded)
    thinkplot.plot(abs(fft_window), color='0.7', label='filter')

    thinkplot.config(xlabel='frequency (1/days)',
                     xlim=[0, 1650/2],
                     ylabel='amplitude ratio',
                     ylim=[0, 4],
                     loc='upper left')
    thinkplot.save(root='systems5')


def plot_filters(wave):

    window1 = numpy.array([1, -1])
    window2 = numpy.array([-1, 4, -3]) / 2.0
    window3 = numpy.array([2, -9, 18, -11]) / 6.0
    window4 = numpy.array([-3, 16, -36, 48, -25]) / 12.0
    window5 = numpy.array([12, -75, 200, -300, 300, -137]) / 60.0

    thinkplot.preplot(5)
    for i, window in enumerate([window1, window2, window3, window4, window5]):
        padded = zero_pad(window, len(wave))
        fft_window = numpy.fft.rfft(padded)
        n = len(fft_window)
        thinkplot.plot(abs(fft_window)[:], label=i+1)

    thinkplot.show()


def plot_response():
    thinkplot.preplot(cols=2)

    response = thinkdsp.read_wave('180961__kleeb__gunshots.wav')
    response = response.segment(start=0.26, duration=5.0)
    response.normalize()
    response.plot()
    thinkplot.config(xlabel='time',
                     xlim=[0, 5.0],
                     ylabel='amplitude',
                     ylim=[-1.05, 1.05])

    thinkplot.subplot(2)
    transfer = response.make_spectrum()
    transfer.plot()
    thinkplot.config(xlabel='frequency',
                     xlim=[0, 22500],
                     ylabel='amplitude')

    thinkplot.save(root='systems6')

    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')
    wave.ys = wave.ys[:len(response)]
    wave.normalize()
    spectrum = wave.make_spectrum()

    output = (spectrum * transfer).make_wave()
    output.normalize()

    wave.plot(color='0.7')
    output.plot(alpha=0.4)
    thinkplot.config(xlabel='time',
                     xlim=[0, 5.0],
                     ylabel='amplitude',
                     ylim=[-1.05, 1.05])
    thinkplot.save(root='systems7')


def shifted_scaled(wave, shift, factor):
    res = wave.copy()
    res.shift(shift)
    res.scale(factor)
    return res

def plot_convolution():
    response = thinkdsp.read_wave('180961__kleeb__gunshots.wav')
    response = response.segment(start=0.26, duration=5.0)
    response.normalize()

    dt = 1
    shift = dt * response.framerate
    factor = 0.5
    
    gun2 = response + shifted_scaled(response, shift, factor)
    gun2.plot()
    thinkplot.config(xlabel='time (s)',
                     ylabel='amplitude',
                     ylim=[-1.05, 1.05],
                     legend=False)
    thinkplot.save(root='systems8')

    signal = thinkdsp.SawtoothSignal(freq=410)
    wave = signal.make_wave(duration=0.1, framerate=response.framerate)

    total = 0
    for j, y in enumerate(wave.ys):
        total += shifted_scaled(response, j, y)

    total.normalize()

    wave.make_spectrum().plot(high=500, color='0.7', label='original')
    segment = total.segment(duration=0.2)
    segment.make_spectrum().plot(high=1000, label='convolved')
    thinkplot.config(xlabel='frequency (Hz)', ylabel='amplitude')
    thinkplot.save(root='systems9')


def main():
    plot_convolution()
    return

    plot_response()
    return

    df = pandas.read_csv('coindesk-bpi-USD-close.csv', 
                         nrows=1625,
                         parse_dates=[0])
    ys = df.Close.values
    wave = thinkdsp.Wave(ys, framerate=1)
    #plot_wave_and_spectrum(wave, root='systems1')

    diff = numpy.diff(ys)
    wave2 = thinkdsp.Wave(diff, framerate=1)
    #plot_wave_and_spectrum(wave2, root='systems2')

    #plot_ratios(wave, wave2)
    plot_derivative(wave, wave2)
    return

    plot_filters(wave)


if __name__ == '__main__':
    main()
