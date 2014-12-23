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
    """Synthesize a mixture of complex sinusoids with given amps and freqs.

    amps: amplitudes
    freqs: frequencies in Hz
    ts: times to evaluate the signal

    returns: wave array
    """
    components = [thinkdsp.ComplexSinusoid(freq, amp)
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
    i = complex(0, 1)
    args = numpy.outer(ts, freqs)
    M = numpy.exp(i * PI2 * args)
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
    M = numpy.exp(i * PI2 * args)
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


def idft(ys):
    """Computes the inverse DFT.

    ys: wave array

    returns: vector of amplitudes
    """
    i = complex(0, 1)
    N = len(ys)
    ts = numpy.arange(N) / N
    freqs = numpy.arange(N)
    args = numpy.outer(ts, freqs)
    M = numpy.exp(i * PI2 * args)
    amps = numpy.dot(M, ys)
    return amps


def make_figures():
    amps = numpy.array([0.6, 0.25, 0.1, 0.05])
    freqs = [100, 200, 300, 400]
    framerate = 11025

    ts = numpy.linspace(0, 1, framerate)
    ys = synthesize1(amps, freqs, ts)
    print(ys)
    
    thinkplot.preplot(2)
    n = framerate / 25
    thinkplot.plot(ts[:n], ys[:n].real, label='real')
    thinkplot.plot(ts[:n], ys[:n].imag, label='imag')
    thinkplot.save(root='dft1',
                   xlabel='time (s)',
                   ylabel='wave',
                   ylim=[-1.05, 1.05])

    ys = synthesize2(amps, freqs, ts)

    amps2 = amps * numpy.exp(1j)
    ys2 = synthesize2(amps2, freqs, ts)

    thinkplot.preplot(2)
    thinkplot.plot(ts[:n], ys.real[:n], label=r'$\phi_0 = 0$')
    thinkplot.plot(ts[:n], ys2.real[:n], label=r'$\phi_0 = 1$')
    thinkplot.save(root='dft2',
                   xlabel='time (s)', 
                   ylabel='wave',
                   ylim=[-1.05, 1.05],
                   loc='lower right')


def main():
    numpy.set_printoptions(precision=3, suppress=True)

    make_figures()

    test1()
    test2()


if __name__ == '__main__':
    main()
