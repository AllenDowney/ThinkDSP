"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot
import thinkstats2
import numpy as np

PI2 = 2 * np.pi


def make_sine(offset):
    """Makes a 440 Hz sine wave with the given phase offset.

    offset: phase offset in radians

    returns: Wave objects
    """
    signal = thinkdsp.SinSignal(freq=440, offset=offset)
    wave = signal.make_wave(duration=0.5, framerate=10000)
    return wave


def corrcoef(xs, ys):
    """Coefficient of correlation.

    ddof=0 indicates that we should normalize by N, not N-1.

    xs: sequence
    ys: sequence

    returns: float
    """
    return np.corrcoef(xs, ys, ddof=0)[0, 1]


def plot_sines():
    """Makes figures showing correlation of sine waves with offsets.
    """
    wave1 = make_sine(0)
    wave2 = make_sine(offset=1)

    thinkplot.preplot(2)
    wave1.segment(duration=0.01).plot(label='wave1')
    wave2.segment(duration=0.01).plot(label='wave2')

    corr_matrix = np.corrcoef(wave1.ys, wave2.ys, ddof=0)
    print(corr_matrix)

    thinkplot.save(root='autocorr1',
                   xlabel='Time (s)',
                   ylim=[-1.05, 1.05])

    offsets = np.linspace(0, PI2, 101)

    corrs = []
    for offset in offsets:
        wave2 = make_sine(offset)
        corr = corrcoef(wave1.ys, wave2.ys)
        corrs.append(corr)
    
    thinkplot.plot(offsets, corrs)
    thinkplot.save(root='autocorr2',
                   xlabel='Offset (radians)',
                   ylabel='Correlation',
                   xlim=[0, PI2],
                   ylim=[-1.05, 1.05])


def plot_shifted(wave, offset=0.002, start=0.2):
    """Plots two segments of a wave with different start times.

    wave: Wave
    offset: difference in start time (seconds)
    start: start time in seconds
    """
    thinkplot.preplot(num=2)
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    segment2 = wave.segment(start=start-offset, duration=0.01)
    segment2.shift(offset)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    thinkplot.text(start+0.0005, -0.8, text)
    thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])


def serial_corr(wave, lag=1):
    """Computes serial correlation with given lag.

    wave: Wave
    lag: integer, how much to shift the wave

    returns: float correlation coefficient
    """
    n = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:n-lag]
    corr = corrcoef(y1, y2)
    return corr


def plot_serial_corr():
    """Makes a plot showing serial correlation for pink noise.
    """
    np.random.seed(19)

    betas = np.linspace(0, 2, 21)
    corrs = []

    for beta in betas:
        signal = thinkdsp.PinkNoise(beta=beta)
        wave = signal.make_wave(duration=1.0, framerate=11025)
        corr = serial_corr(wave)
        corrs.append(corr)

    thinkplot.plot(betas, corrs)
    thinkplot.config(xlabel=r'Pink noise parameter, $\beta$',
                     ylabel='Serial correlation', 
                     ylim=[0, 1.05])
    thinkplot.save(root='autocorr3')


def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    """
    lags = range(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs


def plot_pink_autocorr(beta, label):
    """Makes a plot showing autocorrelation for pink noise.

    beta: parameter of pink noise
    label: string label for the plot
    """
    signal = thinkdsp.PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=11025)
    lags, corrs = autocorr(wave)
    thinkplot.plot(lags, corrs, label=label)


def plot_autocorr():
    """Plots autocorrelation for pink noise with different parameters
    """
    np.random.seed(19)
    thinkplot.preplot(3)

    for beta in [1.7, 1.0, 0.3]:
        label = r'$\beta$ = %.1f' % beta
        plot_pink_autocorr(beta, label)

    thinkplot.config(xlabel='Lag', 
                     ylabel='Correlation', 
                     xlim=[-5, 1000], 
                     ylim=[-0.05, 1.05])
    thinkplot.save(root='autocorr4')


def plot_singing_chirp():
    """Makes a spectrogram of the vocal chirp recording.
    """
    wave = thinkdsp.read_wave('28042__bcjordan__voicedownbew.wav')
    wave.normalize()

    duration = 0.01
    segment = wave.segment(start=0.2, duration=duration)

    # plot two copies of the wave with a shift
    plot_shifted(wave, start=0.2, offset=0.0023)
    thinkplot.save(root='autocorr7')

    # plot the autocorrelation function
    lags, corrs = autocorr(segment)
    thinkplot.plot(lags, corrs)
    thinkplot.config(xlabel='Lag (index)', 
                     ylabel='Correlation', 
                     ylim=[-1.05, 1.05],
                     xlim=[0, 225])
    thinkplot.save(root='autocorr8')

    # plot the spectrogram
    gram = wave.make_spectrogram(seg_length=1024)
    gram.plot(high=4200)

    thinkplot.config(xlabel='Time (s)', 
                     ylabel='Frequency (Hz)',
                     xlim=[0, 1.4],
                     ylim=[0, 4200])
    thinkplot.save(root='autocorr5')
    
    # plot the spectrum of one segment
    spectrum = segment.make_spectrum()
    spectrum.plot(high=1000)
    thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')
    thinkplot.save(root='autocorr6')


def plot_correlate():
    """Plots the autocorrelation function computed by np.
    """
    wave = thinkdsp.read_wave('28042__bcjordan__voicedownbew.wav')
    wave.normalize()
    segment = wave.segment(start=0.2, duration=0.01)

    lags, corrs = autocorr(segment)

    N = len(segment)
    corrs2 = np.correlate(segment.ys, segment.ys, mode='same')
    lags = np.arange(-N//2, N//2)
    thinkplot.plot(lags, corrs2)
    thinkplot.config(xlabel='Lag', 
                     ylabel='Correlation', 
                     xlim=[-N//2, N//2])
    thinkplot.save(root='autocorr9')


def main():
    plot_sines()
    plot_serial_corr()
    plot_autocorr()
    plot_singing_chirp()
    plot_correlate()


if __name__ == '__main__':
    main()
