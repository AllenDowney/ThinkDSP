"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math
import numpy
import scipy.signal
import pandas

import thinkdsp
import thinkplot


PI2 = math.pi * 2


def plot_bitcoin():
    nrows = 1625
    df = pandas.read_csv('coindesk-bpi-USD-close.csv', 
                         nrows=nrows, parse_dates=[0])
    ys = df.Close.values
    diff = numpy.diff(ys)
    changes = 100 * diff / ys[:len(diff)]

    window = scipy.signal.boxcar(30)
    window /= sum(window)
    smoothed = numpy.convolve(changes, window, mode='same')

    thinkplot.plot(changes, color='0.7', label='daily')
    thinkplot.plot(smoothed, label='30 day average')
    thinkplot.config(xlabel='time (days)', 
                     ylabel='% change',
                     xlim=[0, nrows],
                     ylim=[-60, 60],
                     loc='lower right')
    thinkplot.save(root='convolution1')


def plot_boxcar():
    """
    """
    # start with a square signal
    signal = thinkdsp.SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)

    # and a boxcar window
    window = numpy.ones(11)
    window /= sum(window)

    # select a short segment of the wave
    segment = wave.segment(duration=0.01)

    # and pad with window out to the length of the array
    padded = numpy.zeros_like(segment.ys)
    padded[:len(window)] = window

    # compute the first element of the smoothed signal
    prod = padded * segment.ys
    print(sum(prod))

    # compute the rest of the smoothed signal
    smoothed = numpy.zeros_like(segment.ys)
    rolled = padded
    for i in range(len(segment.ys)):
        smoothed[i] = sum(rolled * segment.ys)
        rolled = numpy.roll(rolled, 1)

    # plot the results
    segment.plot(color='0.7')
    smooth = thinkdsp.Wave(smoothed, framerate=wave.framerate)
    smooth.plot()
    thinkplot.config(ylim=[-1.05, 1.05], legend=False)
    thinkplot.save(root='convolution2')

    # compute the same thing using numpy.convolve
    segment.plot(color='0.7')
    ys = numpy.convolve(segment.ys, window, mode='valid')
    smooth2 = thinkdsp.Wave(ys, framerate=wave.framerate)
    smooth2.plot()
    thinkplot.config(ylim=[-1.05, 1.05], legend=False)
    thinkplot.save(root='convolution3')

    # plot the spectrum before and after smoothing
    spectrum = wave.make_spectrum()
    spectrum.plot(color='0.7')

    ys = numpy.convolve(wave.ys, window, mode='same')
    smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()
    spectrum2.plot()
    thinkplot.config(xlabel='frequency (Hz)',
                     ylabel='amplitude',
                     xlim=[0, 22050], 
                     legend=False)
    thinkplot.save(root='convolution4')

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0
    thinkplot.plot(ratio)

    thinkplot.config(xlabel='frequency (Hz)',
                     ylabel='amplitude ratio',
                     xlim=[0, 22050], 
                     legend=False)
    thinkplot.save(root='convolution5')


    # plot the same ratio along with the FFT of the window
    padded = numpy.zeros_like(wave.ys)
    padded[:len(window)] = window
    dft_window = numpy.fft.rfft(padded)

    thinkplot.plot(abs(dft_window), color='0.7', label='boxcar filter')
    thinkplot.plot(ratio, label='amplitude ratio')

    thinkplot.config(xlabel='frequency (Hz)',
                     ylabel='amplitude ratio',
                     xlim=[0, 22050], 
                     legend=False)
    thinkplot.save(root='convolution6')

    
def plot_gaussian():
    """
    """
    # start with a square signal
    signal = thinkdsp.SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)
    spectrum = wave.make_spectrum()

    # and a boxcar window
    boxcar = numpy.ones(11)
    boxcar /= sum(boxcar)

    # and a gaussian window
    gaussian = scipy.signal.gaussian(M=11, std=2)
    gaussian /= sum(gaussian)

    thinkplot.preplot(2)
    thinkplot.plot(boxcar, label='boxcar')
    thinkplot.plot(gaussian, label='Gaussian')
    thinkplot.config(xlabel='index',
                     ylabel='amplitude')
    thinkplot.save(root='convolution7')

    ys = numpy.convolve(wave.ys, gaussian, mode='same')
    smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0

    # plot the same ratio along with the FFT of the window
    padded = numpy.zeros_like(wave.ys)
    padded[:len(gaussian)] = gaussian
    dft_gaussian = numpy.fft.rfft(padded)

    thinkplot.plot(abs(dft_gaussian), color='0.7', label='Gaussian filter')
    thinkplot.plot(ratio, label='amplitude ratio')

    thinkplot.config(xlabel='frequency (Hz)',
                     ylabel='amplitude ratio',
                     xlim=[0, 22050], 
                     legend=False)
    thinkplot.save(root='convolution8')

    
def main():
    plot_gaussian()
    plot_boxcar()
    plot_bitcoin()


if __name__ == '__main__':
    main()
