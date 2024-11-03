"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot

import numpy as np


def plot_filter():
    """Plots the filter that corresponds to a 2-element moving average.
    """
    impulse = np.zeros(8)
    impulse[0] = 1
    wave = thinkdsp.Wave(impulse, framerate=8)
    
    impulse_spectrum = wave.make_spectrum(full=True)
    window_array = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0,])
    window = thinkdsp.Wave(window_array, framerate=8)
    filtr = window.make_spectrum(full=True)

    filtr.plot()
    thinkplot.config(xlabel='Frequency', ylabel='Amplitude')
    thinkplot.save('systems1')


def read_response():
    """Reads the impulse response file and removes the initial silence.
    """
    response = thinkdsp.read_wave('180960__kleeb__gunshot.wav')
    start = 0.26
    response = response.segment(start=start)
    response.shift(-start)
    response.normalize()
    return response


def plot_response(response):
    """Plots an input wave and the corresponding output.
    """
    thinkplot.preplot(cols=2)
    response.plot()
    thinkplot.config(xlabel='Time (s)',
                     xlim=[0.26, response.end],
                     ylim=[-1.05, 1.05])

    thinkplot.subplot(2)
    transfer = response.make_spectrum()
    transfer.plot()
    thinkplot.config(xlabel='Frequency (Hz)',
                     xlim=[0, 22500],
                     ylabel='Amplitude')
    thinkplot.save(root='systems6')

    violin = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    start = 0.11
    violin = violin.segment(start=start)
    violin.shift(-start)

    violin.truncate(len(response))
    violin.normalize()
    spectrum = violin.make_spectrum()

    output = (spectrum * transfer).make_wave()
    output.normalize()

    thinkplot.preplot(rows=2)
    violin.plot(label='input')
    thinkplot.config(xlim=[0, violin.end],
                     ylim=[-1.05, 1.05])

    thinkplot.subplot(2)
    output.plot(label='output')
    thinkplot.config(xlabel='Time (s)',
                     xlim=[0, violin.end],
                     ylim=[-1.05, 1.05])

    thinkplot.save(root='systems7')


def shifted_scaled(wave, shift, factor):
    res = wave.copy()
    res.shift(shift)
    res.scale(factor)
    return res


def plot_convolution(response):
    """Plots the impulse response and a shifted, scaled copy.
    """
    shift = 1
    factor = 0.5
    
    gun2 = response + shifted_scaled(response, shift, factor)
    gun2.plot()
    thinkplot.config(xlabel='Time (s)',
                     ylim=[-1.05, 1.05],
                     legend=False)
    thinkplot.save(root='systems8')


def plot_sawtooth(response):
    signal = thinkdsp.SawtoothSignal(freq=441)
    wave = signal.make_wave(duration=0.1, framerate=response.framerate)

    total = 0
    for t, y in zip(wave.ts, wave.ys):
        total += shifted_scaled(response, t, y)

    total.normalize()

    high = 5000
    wave.make_spectrum().plot(high=high, color='0.7', label='original')
    segment = total.segment(duration=0.2)
    segment.make_spectrum().plot(high=high, label='convolved')
    thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')
    thinkplot.save(root='systems9')


def main():
    plot_filter()

    response = read_response()
    plot_response(response)
    plot_convolution(response)


if __name__ == '__main__':
    main()
