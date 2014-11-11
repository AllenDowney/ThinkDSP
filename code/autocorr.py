"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot
import thinkstats2
import math
import numpy

PI2 = 2 * math.pi


def make_wave(offset):
    signal = thinkdsp.SinSignal(freq=440, offset=offset)
    wave = signal.make_wave(duration=0.5, framerate=10000)
    return wave


def make_figures():

    wave1 = make_wave(0)
    wave2 = make_wave(offset=1)

    thinkplot.preplot(2)
    wave1.segment(duration=0.01).plot(label='wave1')
    wave2.segment(duration=0.01).plot(label='wave2')

    numpy.corrcoef(wave1.ys, wave2.ys)

    thinkplot.save(root='autocorr1',
                   xlabel='time (s)',
                   ylabel='amplitude')


    offsets = numpy.linspace(0, PI2, 101)

    corrs = []
    for offset in offsets:
        wave2 = make_wave(offset)
        corr = numpy.corrcoef(wave1.ys, wave2.ys)[0, 1]
        corrs.append(corr)
    
    thinkplot.plot(offsets, corrs)
    thinkplot.save(root='autocorr2',
                   xlabel='offset (radians)',
                   ylabel='correlation',
                   xlim=[0, PI2])


def plot_shifted(wave, shift=0.002, start=0.2):

    thinkplot.preplot(num=2)
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    segment2 = wave.segment(start=start-shift, duration=0.01)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    thinkplot.text(0.0005, -0.8, text)


def track_pitch(wave, seg_length=512):

    n = len(wave.ys)
    # window = window_func(seg_length)

    start, end, step = 0, seg_length, seg_length / 2

    while end < n:
        ys = wave.ys[start:end]
        segment = thinkdsp.Wave(ys, wave.framerate)

        #segment.plot()
        #thinkplot.show()

        autocorr(segment)
        #correlate(segment)
        break

        t = (start + end) / 2.0 / wave.framerate
        # spec_map[t] = Spectrum(hs, wave.framerate)

        start += step
        end += step


def autocorr(wave):
    n = len(wave.ys)

    corrs = []
    lags = range(n//2)
    for lag in lags:
        y1 = wave.ys[lag:]
        y2 = wave.ys[:n-lag]
        corr = numpy.corrcoef(y1, y2)[0, 1]
        corrs.append(corr)

    thinkplot.plot(lags, corrs)
    thinkplot.show()


def correlate(wave):
    corrs = numpy.correlate(wave.ys, wave.ys, mode='valid')
    print(len(corrs))
    thinkplot.plot(corrs)
    thinkplot.show()

def main():
    
    # make_figures()

    wave = thinkdsp.read_wave('28042__bcjordan__voicedownbew.wav')
    wave.unbias()
    wave.normalize()
    track_pitch(wave)

    
    return


    thinkplot.preplot(rows=1, cols=2)
    plot_shifted(wave, 0.0003)
    thinkplot.config(xlabel='time (s)',
                     ylabel='amplitude',
                     ylim=[-1, 1])

    thinkplot.subplot(2)
    plot_shifted(wave, 0.00225)
    thinkplot.config(xlabel='time (s)',
                     ylim=[-1, 1])

    thinkplot.save(root='autocorr3')


if __name__ == '__main__':
    main()
