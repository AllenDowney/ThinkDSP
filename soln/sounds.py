"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot

FORMATS = ['pdf', 'eps']


def plot_tuning(start=7.0, duration=0.006835):
    """Plots three cycles of a bell playing A4.

    start: start time in seconds
    duration: float
    """
    period = duration/3
    freq = 1/period
    print(period, freq)
    assert abs(freq - 438.917337235) < 1e-7

    wave = thinkdsp.read_wave('18871__zippi1__sound-bell-440hz.wav')

    segment = wave.segment(start, duration)
    segment.normalize()

    thinkplot.preplot(1)
    segment.plot()
    thinkplot.Save(root='sounds1',
                   xlabel='Time (s)',
                   axis=[start, start+duration, -1.05, 1.05],
                   formats=FORMATS,
                   legend=False)


def plot_violin(start=1.30245, duration=0.00683):
    """Plots three cycles of a violin playing A4.

    duration: float
    """
    period = duration/3
    freq = 1/period
    print(period, freq)
    assert abs(freq - 439.238653001) < 1e-7

    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    segment = wave.segment(start, duration)
    segment.normalize()

    thinkplot.preplot(1)
    segment.plot()
    thinkplot.Save(root='sounds2',
                   xlabel='Time (s)',
                   axis=[start, start+duration, -1.05, 1.05],
                   formats=FORMATS,
                   legend=False)


def segment_violin(start=1.2, duration=0.6):
    """Load a violin recording and plot its spectrum.

    start: start time of the segment in seconds
    duration: in seconds
    """
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    # extract a segment
    segment = wave.segment(start, duration)
    segment.normalize()

    # plot the spectrum
    spectrum = segment.make_spectrum()

    thinkplot.preplot(1)
    spectrum.plot(high=10000)
    thinkplot.Save(root='sounds3',
                   xlabel='Frequency (Hz)',
                   ylabel='Amplitude',
                   formats=FORMATS,
                   legend=False)

    # print the top 5 peaks
    peaks = spectrum.peaks()
    for amp, freq in peaks[:10]:
        print(freq, amp)
    assert abs(peaks[0][0] - 3762.382899) < 1e-7


def mix_cosines():
    """Plots three periods of a mix of cosines.
    """

    # create a SumSignal
    cos_sig = thinkdsp.CosSignal(freq=440, amp=1.0, offset=0)
    sin_sig = thinkdsp.SinSignal(freq=880, amp=0.5, offset=0)

    mix = sin_sig + cos_sig

    # create a wave
    wave = mix.make_wave(duration=1.0, start=0, framerate=11025)
    print('Number of samples', len(wave))
    print('Timestep in ms', 1000 / wave.framerate)
    assert len(wave) == wave.framerate

    # select a segment
    period = mix.period
    segment = wave.segment(start=0, duration=period*3)

    # plot the segment
    thinkplot.preplot(1)
    segment.plot()
    thinkplot.Save(root='sounds4',
                   xlabel='Time (s)',
                   axis=[0, period*3, -1.55, 1.55],
                   formats=FORMATS,
                   legend=False)


def main():
    plot_tuning()
    plot_violin()
    segment_violin()
    mix_cosines()


if __name__ == '__main__':
    main()
