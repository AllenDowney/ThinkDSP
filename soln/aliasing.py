"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot

FORMATS = ['pdf', 'png']

def triangle_example(freq):
    """Makes a figure showing a triangle wave.

    freq: frequency in Hz
    """
    framerate = 10000
    signal = thinkdsp.TriangleSignal(freq)

    duration = signal.period*3
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot()
    thinkplot.save(root='triangle-%d-1' % freq,
                   xlabel='Time (s)',
                   axis=[0, duration, -1.05, 1.05])

    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()

    thinkplot.preplot(cols=2)
    spectrum.plot()
    thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude')

    thinkplot.subplot(2)
    spectrum.plot()
    thinkplot.config(ylim=[0, 500],
                     xlabel='Frequency (Hz)')
    
    thinkplot.save(root='triangle-%d-2' % freq)


def square_example(freq):
    """Makes a figure showing a square wave.

    freq: frequency in Hz
    """
    framerate = 10000
    signal = thinkdsp.SquareSignal(freq)

    duration = signal.period*3
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot()
    thinkplot.save(root='square-%d-1' % freq,
                   xlabel='Time (s)',
                   axis=[0, duration, -1.05, 1.05])

    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot()
    thinkplot.save(root='square-%d-2' % freq,
                   xlabel='Frequency (Hz)',
                   ylabel='Amplitude')


def aliasing_example(offset=0.000003):
    """Makes a figure showing the effect of aliasing.
    """
    framerate = 10000

    def plot_segment(freq):
        signal = thinkdsp.CosSignal(freq)
        duration = signal.period*4
        thinkplot.Hlines(0, 0, duration, color='gray')
        segment = signal.make_wave(duration, framerate=framerate*10)
        segment.plot(linewidth=0.5, color='gray')
        segment = signal.make_wave(duration, framerate=framerate)
        segment.plot_vlines(label=freq, linewidth=4)

    thinkplot.preplot(rows=2)
    plot_segment(4500)
    thinkplot.config(axis=[-0.00002, 0.0007, -1.05, 1.05])

    thinkplot.subplot(2)
    plot_segment(5500)
    thinkplot.config(axis=[-0.00002, 0.0007, -1.05, 1.05])

    thinkplot.save(root='aliasing1',
                   xlabel='Time (s)',
                   formats=FORMATS)


def main():
    triangle_example(freq=200)
    triangle_example(freq=1100)
    square_example(freq=100)
    aliasing_example()


if __name__ == '__main__':
    main()
