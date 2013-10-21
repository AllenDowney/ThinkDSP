"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy
import thinkbayes
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
    print len(segment.ys)

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


def test_noise(signal, root):
    wave = signal.make_wave(duration=1.0, framerate=32768)
    # wave.play()

    segment = wave
    segment.plot()
    thinkplot.save(root=root + '1',
                   xlabel='time (s)',
                   ylabel='amplitude')

    spectrum = segment.make_spectrum()
    spectrum.plot(low=1, exponent=2)
    thinkplot.save(root=root + '2',
                   xlabel='frequency (Hz)',
                   ylabel='power',
                   xscale='log',
                   yscale='log')

    integ = spectrum.make_integrated_spectrum()
    integ.plot(low=1, complement=False)
    thinkplot.save(root=root + '3',
                   xlabel='frequency (Hz)',
                   ylabel='cumulative power',
                   xscale='linear',
                   yscale='linear')
    
    return spectrum, integ


def make_periodogram(signal):
    specs = []
    for i in range(1000):
        wave = signal.make_wave(duration=1.0, framerate=32768)
        spec = wave.make_spectrum()
        specs.append(spec)
    
    spectrum = sum(specs)
    print spectrum.estimate_slope()
    spectrum.plot(exponent=2)
    thinkplot.show(xlabel='frequency (Hz)',
                   ylabel='power',
                   xscale='log',
                   yscale='log')


def white_noise():
    framerate = 11025
    duration = 0.5
    n = framerate * duration
    ys = numpy.random.uniform(-1, 1, n)
    wave = thinkdsp.Wave(ys, framerate)

    segment = wave.segment(duration=0.1)
    segment.plot(linewidth=1, alpha=0.5)
    thinkplot.save(root='noise0',
                   xlabel='time (s)',
                   ylabel='amplitude')

    spectrum = wave.make_spectrum()
    spectrum.plot(linewidth=1, alpha=0.5)
    thinkplot.save(root='noise1',
                   xlabel='frequency (Hz)',
                   ylabel='amplitude')

    integ = spectrum.make_integrated_spectrum()
    integ.plot()
    thinkplot.save(root='noise2',
                   xlabel='frequency (Hz)',
                   ylabel='normalized power')

    cdf = thinkbayes.MakeCdfFromList(spectrum.power)
    thinkplot.cdf(cdf, complement=True)
    thinkplot.save(root='noise3',
                   xlabel='power',
                   ylabel='CDF',
                   yscale='log')



def main():
    white_noise()
    return

    thinkdsp.random_seed(19)

    signal = thinkdsp.WhiteNoise()
    white, integ_white = test_noise(signal, 'white-noise')
    print white.estimate_slope()

    signal = thinkdsp.BrownianNoise()
    red, integ_red = test_noise(signal, 'red-noise')
    print red.estimate_slope()
    return

    signal = thinkdsp.PinkNoise(beta=1.0)
    pink, integ_pink = test_noise(signal, 'pink-noise')
    print pink.estimate_slope()

    thinkplot.preplot(num=3)
    white.plot(low=1, exponent=2, label='white', linewidth=2)
    pink.plot(low=1, exponent=2, label='pink', linewidth=2)
    red.plot(low=1, exponent=2, label='red', linewidth=2)
    thinkplot.show(xlabel='frequency (Hz)',
                   ylabel='power',
                   xscale='log',
                   yscale='log',
                   axis=[1, 18000, 1e-5, 1e8])
    

    return
    signal = thinkdsp.BrownianNoise()
    make_periodogram(signal)



if __name__ == '__main__':
    main()
