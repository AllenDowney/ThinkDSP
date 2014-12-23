"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot


def segment_violin(start=1.2, duration=0.6):
    """Load a violin recording and plot its spectrum.

    start: start time of the segment in seconds
    duration: in seconds
    """
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    # extract a segment
    segment = wave.segment(start, duration)
    segment.normalize()
    segment.apodize()
    segment.write('violin_segment1.wav')

    # plot the spectrum
    spectrum = segment.make_spectrum()
    n = len(spectrum.hs)
    spectrum.plot(high=n/2)
    thinkplot.Save(root='violin2',
                   xlabel='frequency (Hz)',
                   ylabel='amplitude density')

    # print the top 5 peaks
    peaks = spectrum.peaks()
    for amp, freq in peaks[:10]:
        print(freq, amp)

    # compare the segments to a 440 Hz Triangle wave
    note = thinkdsp.make_note(69, 0.6, 
                              sig_cons=thinkdsp.TriangleSignal, 
                              framerate=segment.framerate)

    wfile = thinkdsp.WavFileWriter('violin_segment2.wav', note.framerate)
    wfile.write(note)
    wfile.write(segment)
    wfile.write(note)
    wfile.close()


def sin_spectrum():
    """Plots the spectrum of a sine wave.
    """
    wave = thinkdsp.make_note(69, 0.5, SinSignal)
    spectrum = wave.spectrum()
    spectrum.plot()
    thinkplot.Show()

    peaks = spectrum.peaks()
    print(peaks[0])

    wave2 = spectrum.make_wave()

    wave2.plot()
    thinkplot.Show()

    wave2.write()


def plot_sinusoid(duration = 0.00685):
    """Plots three cycles of a 440 Hz sinusoid.

    duration: float
    """
    signal = thinkdsp.SinSignal(440)
    wave = signal.make_wave(duration, framerate=44100)
    wave.plot()
    thinkplot.Save(root='sinusoid1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def plot_violin(start=1.30245, duration=0.00683):
    """Plots three cycles of a violin playing A4.

    duration: float
    """
    period = duration/3
    freq = 1/period
    print(freq)

    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    segment = wave.segment(start, duration)
    segment.normalize()

    segment.plot()
    thinkplot.Save(root='violin1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def plot_tuning(start=7.0, duration=0.006835):
    """Plots three cycles of a tuning fork playing A4.

    start: start time in seconds
    duration: float
    """
    period = duration/3
    freq = 1/period
    print(period, freq)

    wave = thinkdsp.read_wave('18871__zippi1__sound-bell-440hz.wav')

    segment = wave.segment(start, duration)
    segment.normalize()

    segment.plot()
    thinkplot.Save(root='tuning1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def main():
    segment_violin()
    plot_tuning()
    plot_sinusoid()
    plot_violin()



if __name__ == '__main__':
    main()
