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
    """Synthesize a mixture of cosines with given amps and freqs.

    amps: amplitudes
    freqs: frequencies in Hz
    ts: times to evaluate the signal

    returns: wave array
    """
    components = [thinkdsp.CosSignal(freq, amp)
                  for amp, freq in zip(amps, freqs)]
    signal = thinkdsp.SumSignal(*components)

    ys = signal.evaluate(ts)
    return ys


def synthesize2(amps, freqs, ts):
    """Synthesize a mixture of cosines with given amps and freqs.

    amps: amplitudes
    freqs: frequencies in Hz
    ts: times to evaluate the signal

    returns: wave array
    """
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)

    print 'M'
    print M
    ys = numpy.dot(M, amps)

    print 'ys'
    print ys

    print 'Mt M'
    print numpy.dot(M.T, M)

    return ys


def analyze1(ys, freqs, ts):
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)
    amps = numpy.linalg.solve(M, ys)
    return amps


def analyze2(ys, freqs, ts):
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)
    amps = numpy.dot(M, ys) / 2
    return amps


def dct_iv(ys):
    """Computes DCT-IV.

    ys: wave array

    returns: vector of amplitudes
    """
    N = len(ys)
    ts = (0.5 + numpy.arange(N)) / N
    freqs = (0.5 + numpy.arange(N)) / 2
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)
    amps = numpy.dot(M, ys) / 2
    return amps


def inverse_dct_iv(amps):
    """Computes the inverse of DCT-IV.

    amps: vector of amplitudes

    returns: wave array
    """
    return dct_iv(amps) * 2


def plot(res, index):

    slices = [[0, None], [400, 1000], [860, 900]]
    start, end = slices[index]
    xs, ys = zip(*res[start:end])
    thinkplot.plot(xs, ys)
    thinkplot.save(root='dft%d' % index,
                   xlabel='freq (Hz)',
                   ylabel='cov',
                   formats=['png'])


def test_synthesize():
    """Compares the output of synthesize1 and synthesize2.
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    freqs = [100, 200, 300, 400]
    framerate = 11025
    ts = numpy.linspace(0, 1, framerate)
    ys1 = synthesize1(amps, freqs, ts)
    ys2 = synthesize2(amps, freqs, ts)
    print max(abs(ys1 - ys2))


def synthesize_example():
    """Synthesizes a signal with four components and plays it.
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    freqs = [100, 200, 300, 400]

    framerate = 11025
    ts = numpy.linspace(0, 1, 11025)
    ys = synthesize2(amps, freqs, ts)
    wave = thinkdsp.Wave(ys, framerate)
    wave.normalize()
    wave.apodize()
    wave.play()

    n = len(freqs)
    amps2 = analyze1(ys[:n], freqs, ts[:n])
    print amps
    print amps2


def test_analyze1():
    """Tests analyze1.
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    time_unit = 0.001
    ts = numpy.arange(N) / N * time_unit
    max_freq = N / time_unit / 2
    freqs = numpy.arange(N) / N * max_freq
    print 'amps', amps
    print 'ts', ts / time_unit
    print 'freqs', freqs
    ys = synthesize2(amps, freqs, ts)
    amps2 = analyze1(ys, freqs, ts)
    print 'amps', amps
    print 'amps2', amps2


def test_analyze2():
    """Tests analyze2.
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + numpy.arange(N)) / N
    freqs = (0.5 + numpy.arange(N)) / 2
    print 'amps', amps
    print 'ts', ts
    print 'freqs', freqs
    ys = synthesize2(amps, freqs, ts)
    amps2 = analyze2(ys, freqs, ts)
    print 'amps', amps
    print 'amps2', amps2


def test_dct_iv():
    """
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + numpy.arange(N)) / N
    freqs = (0.5 + numpy.arange(N)) / 2
    ys = synthesize2(amps, freqs, ts)

    amps2 = dct_iv(ys)
    print max(abs(amps - amps2))
    


def test_inverse_dct_iv():
    amps = [0.6, 0.25, 0.1, 0.05]
    ys = inverse_dct_iv(amps)
    amps2 = dct_iv(ys)
    print max(abs(amps - amps2))


def test_make_dct():
    N = 32
    framerate = N
    amps = numpy.zeros(N)
    amps[2] = 1.0
    dct = thinkdsp.Dct(amps, framerate)
    wave = dct.make_wave()
    print wave.ys

    cos_sig = thinkdsp.CosSignal(freq=1, offset=math.pi/N)
    wave = cos_sig.make_wave(duration=1, start=0, framerate=framerate)
    print wave.ys

    dct = wave.make_dct()
    dct.plot()
    print dct.fs

    iv = dct_iv(wave.ys)
    thinkplot.plot(dct.fs, iv)
    thinkplot.show()



def main():
    test_inverse_dct_iv()
    return

    test_dct_iv()
    return

    test_synthesize()
    return

    numpy.set_printoptions(precision=3, suppress=True)

    test_make_dct()
    return

    synthesize_example()
    return


    test_synthesize(fshift=0, tshift=0)
    test_synthesize(fshift=0.5, tshift=0.5)
    return

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


if __name__ == '__main__':
    main()
