"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot

import numpy as np
import pandas as pd

import scipy.signal

PI2 = np.pi * 2


def plot_bitcoin():
    """Plot BitCoin prices and a smoothed time series.
    """
    nrows = 1625
    df = pandas.read_csv('coindesk-bpi-USD-close.csv', 
                         nrows=nrows, parse_dates=[0])
    ys = df.Close.values

    window = np.ones(30)
    window /= sum(window)
    smoothed = np.convolve(ys, window, mode='valid')

    N = len(window)
    smoothed = thinkdsp.shift_right(smoothed, N//2)

    thinkplot.plot(ys, color='0.7', label='daily')
    thinkplot.plot(smoothed, label='30 day average')
    thinkplot.config(xlabel='time (days)', 
                     ylabel='price',
                     xlim=[0, nrows],
                     loc='lower right')
    thinkplot.save(root='convolution1')


GRAY = "0.7"

def plot_facebook():
    """Plot Facebook prices and a smoothed time series.
    """
    names = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv('fb.csv', header=0, names=names, parse_dates=[0])
    close = df.close.values[::-1]
    dates = df.date.values[::-1]
    days = (dates - dates[0]) / np.timedelta64(1,'D')

    M = 30
    window = np.ones(M)
    window /= sum(window)
    smoothed = np.convolve(close, window, mode='valid')
    smoothed_days = days[M//2: len(smoothed) + M//2]
    
    thinkplot.plot(days, close, color=GRAY, label='daily close')
    thinkplot.plot(smoothed_days, smoothed, label='30 day average')
    
    last = days[-1]
    thinkplot.config(xlabel='Time (days)', 
                     ylabel='Price ($)',
                     xlim=[-7, last+7],
                     legend=True,
                     loc='lower right')
    thinkplot.save(root='convolution1')


def plot_boxcar():
    """Makes a plot showing the effect of convolution with a boxcar window.
    """
    # start with a square signal
    signal = thinkdsp.SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)

    # and a boxcar window
    window = np.ones(11)
    window /= sum(window)

    # select a short segment of the wave
    segment = wave.segment(duration=0.01)

    # and pad with window out to the length of the array
    N = len(segment)
    padded = thinkdsp.zero_pad(window, N)

    # compute the first element of the smoothed signal
    prod = padded * segment.ys
    print(sum(prod))

    # compute the rest of the smoothed signal
    smoothed = np.zeros(N)
    rolled = padded
    for i in range(N):
        smoothed[i] = sum(rolled * segment.ys)
        rolled = np.roll(rolled, 1)

    # plot the results
    segment.plot(color=GRAY)
    smooth = thinkdsp.Wave(smoothed, framerate=wave.framerate)
    smooth.plot()
    thinkplot.config(xlabel='Time(s)', ylim=[-1.05, 1.05])
    thinkplot.save(root='convolution2')

    # compute the same thing using np.convolve
    segment.plot(color=GRAY)
    ys = np.convolve(segment.ys, window, mode='valid')
    smooth2 = thinkdsp.Wave(ys, framerate=wave.framerate)
    smooth2.plot()
    thinkplot.config(xlabel='Time(s)', ylim=[-1.05, 1.05])
    thinkplot.save(root='convolution3')

    # plot the spectrum before and after smoothing
    spectrum = wave.make_spectrum()
    spectrum.plot(color=GRAY)

    ys = np.convolve(wave.ys, window, mode='same')
    smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()
    spectrum2.plot()
    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude',
                     xlim=[0, 22050])
    thinkplot.save(root='convolution4')

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0
    thinkplot.plot(ratio)

    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050])
    thinkplot.save(root='convolution5')


    # plot the same ratio along with the FFT of the window
    padded = thinkdsp.zero_pad(window, len(wave))
    dft_window = np.fft.rfft(padded)

    thinkplot.plot(abs(dft_window), color=GRAY, label='DFT(window)')
    thinkplot.plot(ratio, label='amplitude ratio')

    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050])
    thinkplot.save(root='convolution6')

    
def plot_gaussian():
    """Makes a plot showing the effect of convolution with a boxcar window.
    """
    # start with a square signal
    signal = thinkdsp.SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)
    spectrum = wave.make_spectrum()

    # and a boxcar window
    boxcar = np.ones(11)
    boxcar /= sum(boxcar)

    # and a gaussian window
    gaussian = scipy.signal.gaussian(M=11, std=2)
    gaussian /= sum(gaussian)

    thinkplot.preplot(2)
    thinkplot.plot(boxcar, label='boxcar')
    thinkplot.plot(gaussian, label='Gaussian')
    thinkplot.config(xlabel='Index', legend=True)
    thinkplot.save(root='convolution7')

    ys = np.convolve(wave.ys, gaussian, mode='same')
    smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0

    # plot the same ratio along with the FFT of the window
    padded = thinkdsp.zero_pad(gaussian, len(wave))
    dft_gaussian = np.fft.rfft(padded)

    thinkplot.plot(abs(dft_gaussian), color=GRAY, label='Gaussian filter')
    thinkplot.plot(ratio, label='amplitude ratio')

    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050])
    thinkplot.save(root='convolution8')


def fft_convolve(signal, window):
    """Computes convolution using FFT.
    """
    fft_signal = np.fft.fft(signal)
    fft_window = np.fft.fft(window)
    return np.fft.ifft(fft_signal * fft_window)


def fft_autocorr(signal):
    """Computes the autocorrelation function using FFT.
    """
    N = len(signal)
    signal = thinkdsp.zero_pad(signal, 2*N)
    window = np.flipud(signal)

    corrs = fft_convolve(signal, window)
    corrs = np.roll(corrs, N//2+1)[:N]
    return corrs


def plot_fft_convolve():
    """Makes a plot showing that FFT-based convolution works.
    """
    names = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv('fb.csv',
                     header=0, names=names, parse_dates=[0])
    close = df.close.values[::-1]

    # compute a 30-day average using np.convolve
    window = scipy.signal.gaussian(M=30, std=6)
    window /= window.sum()
    smoothed = np.convolve(close, window, mode='valid')

    # compute the same thing using fft_convolve
    N = len(close)
    padded = thinkdsp.zero_pad(window, N)

    M = len(window)
    smoothed4 = fft_convolve(close, padded)[M-1:]

    # check for the biggest difference
    diff = smoothed - smoothed4
    print(max(abs(diff)))

    # compute autocorrelation using np.correlate
    corrs = np.correlate(close, close, mode='same')
    corrs2 = fft_autocorr(close)

    # check for the biggest difference
    diff = corrs - corrs2
    print(max(abs(diff)))

    # plot the results
    lags = np.arange(N) - N//2
    thinkplot.plot(lags, corrs, color=GRAY, linewidth=7, label='np.convolve')
    thinkplot.plot(lags, corrs2.real, linewidth=2, label='fft_convolve')
    thinkplot.config(xlabel='Lag', 
                     ylabel='Correlation', 
                     xlim=[-N//2, N//2])
    thinkplot.save(root='convolution9')


def main():
    plot_facebook()
    plot_boxcar()
    plot_gaussian()
    plot_fft_convolve()
    #plot_bitcoin()


if __name__ == '__main__':
    main()
