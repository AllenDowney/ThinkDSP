"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

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
    print spectrum.estimate_slope()
    spectrum.plot_power(low=1)
    thinkplot.show(xlabel='frequency (Hz)',
                   ylabel='power density',
                   xscale='log',
                   yscale='log')


def process_noise(signal, root='red'):
    wave = signal.make_wave(duration=0.5, framerate=11025)

    # 0: waveform
    segment = wave.segment(duration=0.1)
    segment.plot(linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise0',
                   xlabel='time (s)',
                   ylabel='amplitude')

    spectrum = wave.make_spectrum()

    # 1: spectrum
    spectrum.plot_power(linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise1',
                   xlabel='frequency (Hz)',
                   ylabel='power density')

    slope, _, _, _, _ = spectrum.estimate_slope()
    print 'estimated slope', slope

    # 2: integrated spectrum
    integ = spectrum.make_integrated_spectrum()
    integ.plot_power()
    thinkplot.save(root=root+'noise2',
                   xlabel='frequency (Hz)',
                   ylabel='normalized power')

    # 3: log-log spectral density
    spectrum.plot_power(low=1, linewidth=1, alpha=0.5)
    thinkplot.save(root=root+'noise3',
                   xlabel='frequency (Hz)',
                   ylabel='power density',
                   xscale='log',
                   yscale='log')

    # 4: CDF of power density
    cdf = thinkstats2.MakeCdfFromList(spectrum.power)
    thinkplot.cdf(cdf)
    thinkplot.save(root=root+'noise4',
                   xlabel='power density',
                   ylabel='CDF')

    # 5: CCDF of power density, log-y
    thinkplot.cdf(cdf, complement=True)
    thinkplot.save(root=root+'noise5',
                   xlabel='power density',
                   ylabel='log(CCDF)',
                   yscale='log')

    thinkstats2.NormalProbabilityPlot(spectrum.real, label='real',
                                      data_color='#253494')
    thinkstats2.NormalProbabilityPlot(spectrum.imag-50, label='imag-50', 
                                      data_color='#1D91C0')
    thinkplot.save(root=root+'noise6',
                   xlabel='normal sample',
                   ylabel='power density')



def TriplePlot():
    thinkdsp.random_seed(20)

    duration = 1.0
    framerate = 512

    signal = thinkdsp.UncorrelatedUniformNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    white = wave.make_spectrum()

    signal = thinkdsp.PinkNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    pink = wave.make_spectrum()

    signal = thinkdsp.BrownianNoise()
    wave = signal.make_wave(duration=duration, framerate=framerate)
    red = wave.make_spectrum()

    #thinkplot.preplot(num=3)
    linewidth = 1
    white.plot_power(low=1, label='white', color='gray', linewidth=linewidth)
    pink.plot_power(low=1, label='pink', color='pink', linewidth=linewidth)
    red.plot_power(low=1, label='red', color='red', linewidth=linewidth)
    thinkplot.save(root='noise-triple',
                   xlabel='frequency (Hz)',
                   ylabel='power',
                   xscale='log',
                   yscale='log',
                   axis=[1, 300, 1e-4, 1e5]
                   )


def main():
    TriplePlot()
    return

    thinkdsp.random_seed(20)
    signal = thinkdsp.PinkNoise(beta=1.0)
    process_noise(signal, root='pink')
    return

    thinkdsp.random_seed(18)
    signal = thinkdsp.BrownianNoise()
    process_noise(signal, root='red')
    return

    thinkdsp.random_seed(19)
    signal = thinkdsp.BrownianNoise()
    make_periodogram(signal)
    return

    signal = thinkdsp.UncorrelatedUniformNoise()
    wave1 = signal.make_wave(duration=1.0, framerate=11025)
    wave2 = signal.make_wave(duration=1.0, framerate=11025)
    print wave1.cov(wave2)
    print wave1.cov(wave1)
    print wave2.cov(wave2)

    print wave1.cov_mat(wave2)

    print wave1.corr(wave2)
    print wave1.corr(wave1)
    print wave2.corr(wave2)
    return




if __name__ == '__main__':
    main()
