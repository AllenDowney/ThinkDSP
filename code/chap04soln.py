"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkstats2
import thinkdsp
import thinkplot

"""
Samples used in this file

http://www.freesound.org/people/ciccarelli/sounds/132736/download/
132736__ciccarelli__ocean-waves.wav

http://www.freesound.org/people/Q.K./sounds/56311/download/
56311__q-k__rain-06.wav

http://www.freesound.org/people/erkanozan/sounds/51743/download/
51743__erkanozan__applause.wav

http://www.freesound.org/people/britishpirate93/sounds/162313/download/
162313__britishpirate93__16-inch-crash.wav

"""

FILENAMES = [
    '132736__ciccarelli__ocean-waves.wav',
    '56311__q-k__rain-06.wav',
    '51743__erkanozan__applause.wav',
    '162313__britishpirate93__16-inch-crash.wav',
    '75341__neotone__cymbol-scraped.wav',
    '181934__landub__applause2.wav',
    '87594__ohrwurm__occurs-applause.wav',
    '104466__dkmedic__t-flush.wav',
    '180929__docquesting__crowd-noise.wav',
    '695__memexikon__ocarina.wav',
]

STARTS = [0.6, 2.0, 0.1, 0.3, 0.1, 1.0, 11.0, 4.0, 0.5, 0.5] 



def segment_spectrum(filename, start, duration=1.0):
    """Plots the spectrum of a segment of a WAV file.

    Output file names are the given filename plus a suffix.

    filename: string
    start: start time in s
    duration: segment length in s
    """
    wave = thinkdsp.read_wave(filename)
    plot_waveform(wave)

    # extract a segment
    segment = wave.segment(start, duration)
    segment.ys = segment.ys[:1024]
    print(len(segment.ys))

    segment.normalize()
    segment.apodize()
    spectrum = segment.make_spectrum()

    segment.play()

    # plot the spectrum
    n = len(spectrum.hs)
    spectrum.plot()

    thinkplot.save(root=filename,
                   xlabel='frequency (Hz)',
                   ylabel='amplitude density')



def plot_waveform(wave, start=1.30245, duration=0.00683):
    """Plots a short window from a wave.

    duration: float
    """
    segment = wave.segment(start, duration)
    segment.normalize()

    segment.plot()
    thinkplot.save(root='waveform',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def process_files():
    indices = range(len(FILENAMES))
    #indices = [9]

    for i, (filename, start) in enumerate(zip(FILENAMES, STARTS)):
        if i in indices:
            segment_spectrum(filename, start)


def make_periodogram(signal, n=1000):
    """Generates n spectra and plots their average.

    signal: Signal object
    n: number of spectra to average
    """
    specs = []
    for i in range(n):
        wave = signal.make_wave(duration=1.0, framerate=1024)
        spec = wave.make_spectrum()
        specs.append(spec)
    
    spectrum = sum(specs)
    print(spectrum.estimate_slope())
    spectrum.plot_power()
    thinkplot.show(xlabel='frequency (Hz)',
                   ylabel='power density',
                   xscale='log',
                   yscale='log')


def main():
    thinkdsp.random_seed(19)
    signal = thinkdsp.BrownianNoise()
    make_periodogram(signal)


if __name__ == '__main__':
    main()
