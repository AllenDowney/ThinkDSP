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

PI2 = math.pi * 2


def synthesize1(amps, freqs):

    components = [thinkdsp.CosSignal(freq, amp)
                  for amp, freq in zip(amps, freqs)]
    signal = thinkdsp.SumSignal(*components)

    wave = signal.make_wave(duration=1, framerate=len(amps))
    print 'synth1 ys'
    print wave.ys


def synthesize2(amps, freqs):
    N = 1.0 * len(amps)
    ts = (numpy.arange(N) + 0.5) / N
    print 'ts'
    print ts

    args = numpy.outer(ts, freqs)
    print 'args'
    print args
    M = numpy.cos(PI2 * args)

    print 'M'
    print M
    ys = numpy.dot(M, amps)

    print 'ys'
    print ys

    print 'Mt M'
    print numpy.dot(M.T, M)

    print 'dct'
    print numpy.dot(M.T, ys)
    



def main():
    N = 4
    amps = numpy.array([0.5, 0.5, 0, 0])
    freqs = (numpy.arange(N) + 0.5) / 2
    print 'amps', amps
    print 'freqs', freqs

    synthesize1(amps, freqs)
    synthesize2(amps, freqs)
    return

    cos_sig = thinkdsp.CosSignal(freq=1)
    wave = cos_sig.make_wave(duration=1, start=0, framerate=4)
    print wave.ys

    dct = scipy.fftpack.dct(wave.ys, type=2)
    print dct

    cos_trans = wave.cos_transform()
    xs, ys = zip(*cos_trans)
    print ys
    return


    framerate = 4000
    cos_sig = (thinkdsp.CosSignal(freq=440) +
               thinkdsp.SinSignal(freq=660) +
               thinkdsp.CosSignal(freq=880))

    wave = cos_sig.make_wave(duration=0.5, start=0, framerate=framerate)

    res = wave.cos_transform()
    for index in range(3):
        plot(res, index)


def plot(res, index):

    slices = [[0, None], [400, 1000], [860, 900]]
    start, end = slices[index]
    xs, ys = zip(*res[start:end])
    thinkplot.plot(xs, ys)
    thinkplot.save(root='dft%d' % index,
                   xlabel='freq (Hz)',
                   ylabel='cov',
                   formats=['png'])


if __name__ == '__main__':
    main()
