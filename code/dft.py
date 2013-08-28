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


def synthesize1(amps, freqs, ts):

    components = [thinkdsp.CosSignal(freq, amp)
                  for amp, freq in zip(amps, freqs)]
    signal = thinkdsp.SumSignal(*components)

    ys = signal.evaluate(ts)
    print 'synth1 ys'
    print ys
    return ys

def synthesize2(amps, freqs, ts):
    N = 1.0 * len(amps)

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

    return ys



def main():
    numpy.set_printoptions(precision=3, suppress=True)

    N = 4.0
    amps = numpy.array([0.5, 0.25, 0.1, 0.05])
    freqs = numpy.arange(1, N+1) * 110
    ts = numpy.arange(N) / N
    print 'amps', amps
    print 'freqs', freqs
    print 'ts', ts

    ys = synthesize1(amps, freqs, ts)
    ys = synthesize2(amps, freqs, ts)
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
