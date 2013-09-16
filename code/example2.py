"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import numpy
import scipy.fftpack

import thinkdsp
import thinkplot

def triangle_example(freq):
    framerate = 10000
    signal = thinkdsp.TriangleSignal(freq)

    duration = signal.period*3
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot()
    thinkplot.save(root='triangle-%d-1' % freq,
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])

    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot()
    thinkplot.save(root='triangle-%d-2' % freq,
                   xlabel='frequency (Hz)',
                   ylabel='amplitude')

    #spectrum.plot()
    #thinkplot.show(xscale='log', yscale='log')


def square_example(freq):
    framerate = 10000
    signal = thinkdsp.SquareSignal(freq)

    duration = signal.period*3
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot()
    thinkplot.save(root='square-%d-1' % freq,
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])

    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot()
    thinkplot.save(root='square-%d-2' % freq,
                   xlabel='frequency (Hz)',
                   ylabel='amplitude')


def aliasing_example():
    framerate = 10000
    thinkplot.preplot(num=2)

    freq1 = 4500
    signal = thinkdsp.CosSignal(freq1)
    duration = signal.period*5
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot(label=freq1)

    freq2 = 5500
    signal = thinkdsp.CosSignal(freq2)
    #duration = signal.period*10
    segment = signal.make_wave(duration, framerate=framerate)
    segment.plot(label=freq2)

    thinkplot.save(root='aliasing-3',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05]
                   )


def main():
    triangle_example(freq=200)
    triangle_example(freq=1100)
    square_example(freq=100)
    square_example(freq=1100)
    aliasing_example()


if __name__ == '__main__':
    main()
