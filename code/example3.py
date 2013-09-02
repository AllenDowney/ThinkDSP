"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import numpy

import thinkdsp
import thinkplot

import matplotlib.pyplot as pyplot

PI2 = math.pi * 2


def linear_chirp_evaluate(ts, low=440, high=880, amp=1.0):
    print 'ts', ts

    freqs = numpy.linspace(low, high, len(ts)-1)
    print 'freqs', freqs

    dts = numpy.diff(ts)
    print 'dts', dts

    dphis = numpy.insert(PI2 * freqs * dts, 0, 0)
    print 'dphis', dphis

    phases = numpy.cumsum(dphis)
    print 'phases', phases

    ys = amp * numpy.cos(phases)
    print 'ys', ys

    return ys


def discontinuity(num_periods=30, hamming=False):
    signal = thinkdsp.SinSignal(freq=440)
    duration = signal.period * num_periods
    wave = signal.make_wave(duration)

    if hamming:
        wave.hamming()

    print len(wave.ys), wave.ys[0], wave.ys[-1]
    spectrum = wave.make_spectrum()
    spectrum.plot(high=60)



def three_spectrums():    
    thinkplot.preplot(rows=1, cols=3)

    pyplot.subplots_adjust(wspace=0.3, hspace=0.4, 
                           right=0.95, left=0.1,
                           top=0.95, bottom=0.05)

    xticks = xrange(0, 900, 200)

    thinkplot.subplot(1)
    thinkplot.config(xticks=xticks)
    discontinuity(num_periods=30, hamming=False)

    thinkplot.subplot(2)
    thinkplot.config(xticks=xticks)
    discontinuity(num_periods=30.25, hamming=False)

    thinkplot.subplot(3)
    thinkplot.config(xticks=xticks)
    discontinuity(num_periods=30.25, hamming=True)

    thinkplot.save(root='windowing1')


def window_plot():    
    signal = thinkdsp.SinSignal(freq=440)
    duration = signal.period * 10.25
    wave1 = signal.make_wave(duration)
    wave2 = signal.make_wave(duration)

    ys = numpy.hamming(len(wave1.ys))
    window = thinkdsp.Wave(ys, wave1.framerate)

    wave2.hamming()

    thinkplot.preplot(rows=3, cols=1)

    pyplot.subplots_adjust(wspace=0.3, hspace=0.3, 
                           right=0.95, left=0.1,
                           top=0.95, bottom=0.05)

    thinkplot.subplot(1)
    wave1.plot()
    thinkplot.Config(axis=[0, duration, -1.07, 1.07])

    thinkplot.subplot(2)
    window.plot()
    thinkplot.Config(axis=[0, duration, -1.07, 1.07])

    thinkplot.subplot(3)
    wave2.plot()
    thinkplot.Config(axis=[0, duration, -1.07, 1.07],
                     xlabel='time (s)')

    thinkplot.save(root='windowing2')


def chirp_spectrum():
    signal = thinkdsp.Chirp(start=220, end=440)
    wave = signal.make_wave(duration=1)
    spectrum = wave.make_spectrum()
    spectrum.plot(high=660)
    thinkplot.save('chirp1',
                   xlabel='frequency (Hz)',
                   ylabel='amplitude')
    

def chirp_spectrogram():
    signal = thinkdsp.Chirp(start=220, end=440)
    wave = signal.make_wave(duration=1, framerate=11025)
    spectrogram = wave.make_spectrogram(seg_length=512)

    print 'time res', spectrogram.time_res
    print 'freq res', spectrogram.freq_res
    print 'product', spectrogram.time_res * spectrogram.freq_res

    spectrogram.plot(high=32)

    thinkplot.save('chirp2',
                   xlabel='time (s)',
                   ylabel='frequency (Hz)')
    
    
def violin_spectrogram():
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    seg_length = 2048
    spectrogram = wave.make_spectrogram(seg_length)
    spectrogram.plot(high=seg_length/8)

    # TODO: try imshow?

    thinkplot.save('spectrogram1',
                   xlabel='time (s)',
                   ylabel='frequency (Hz)',
                   formats=['pdf'])
    

def overlapping_windows():
    n = 256
    window = numpy.hamming(n)

    thinkplot.preplot(num=5)
    start = 0
    for i in range(5):
        xs = numpy.arange(start, start+n)
        thinkplot.plot(xs, window)

        start += n/2

    thinkplot.show(axis=[0, 800, 0, 1.05])


def main():
    overlapping_windows()
    return

    chirp_spectrogram()
    return

    violin_spectrogram()
    return

    chirp_spectrum()

    window_plot()

    three_spectrums()
    return

    linear_chirp_evaluate(range(4))

    signal = thinkdsp.Chirp(start=220, end=880)
    wave1 = signal.make_wave(duration=2)
    wave1.apodize()

    signal = thinkdsp.ExpoChirp(start=220, end=880)
    wave2 = signal.make_wave(duration=2)
    wave2.apodize()

    filename = 'chirp.wav'
    wfile = thinkdsp.WavFileWriter(filename, wave1.framerate)
    wfile.write(wave1)
    wfile.write(wave2)
    wfile.close
    thinkdsp.play_wave(filename)


if __name__ == '__main__':
    main()
