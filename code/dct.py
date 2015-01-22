"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math
import numpy

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
    ys = numpy.dot(M, amps)
    return ys


def analyze1(ys, freqs, ts):
    """Analyze a mixture of cosines and return amplitudes.

    Works for the general case where M is not orthogonal.

    ys: wave array
    freqs: frequencies in Hz
    ts: times where the signal was evaluated    

    returns: vector of amplitudes
    """
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)
    amps = numpy.linalg.solve(M, ys)
    return amps


def analyze2(ys, freqs, ts):
    """Analyze a mixture of cosines and return amplitudes.

    Assumes that freqs and ts are chosen so that M is orthogonal.

    ys: wave array
    freqs: frequencies in Hz
    ts: times where the signal was evaluated    

    returns: vector of amplitudes
    """
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


def synthesize_example():
    """Synthesizes a sum of sinusoids and plays it.
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
    print(amps)
    print(amps2)


def test1():
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    time_unit = 0.001
    ts = numpy.arange(N) / N * time_unit
    max_freq = N / time_unit / 2
    freqs = numpy.arange(N) / N * max_freq
    args = numpy.outer(ts, freqs)
    M = numpy.cos(PI2 * args)
    return M


def test2():
    """
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + numpy.arange(N)) / N
    freqs = (0.5 + numpy.arange(N)) / 2
    print('amps', amps)
    print('ts', ts)
    print('freqs', freqs)
    ys = synthesize2(amps, freqs, ts)
    amps2 = analyze2(ys, freqs, ts)
    print('amps', amps)
    print('amps2', amps2)


def test_dct():
    """
    """
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + numpy.arange(N)) / N
    freqs = (0.5 + numpy.arange(N)) / 2
    ys = synthesize2(amps, freqs, ts)

    amps2 = dct_iv(ys)
    print('amps', amps)
    print('amps2', amps2)


def main():
    numpy.set_printoptions(precision=3, suppress=True)

    test1()
    test2()


if __name__ == '__main__':
    main()
