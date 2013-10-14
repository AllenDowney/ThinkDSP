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




def test_make_dct(N):
    amps = numpy.zeros(N)
    amps[16] = 1.0
    dct = thinkdsp.Dct(amps, framerate=N)
    wave = dct.make_wave()

    dct = wave.make_dct()
    dct.plot()




def test_dct(N, freq, window=False):
    cos_sig = thinkdsp.CosSignal(freq=freq)

    ts = (0.5 + numpy.arange(N)) / N
    ys = cos_sig.evaluate(ts)
    
    wave = thinkdsp.Wave(ys, framerate=N)

    if window:
        wave.hamming()

    dct = wave.make_dct()
    dct.plot()


def test_fft(N, freq, window=False):
    cos_sig = thinkdsp.CosSignal(freq=freq)

    ts = (0.5 + numpy.arange(N)) / N
    ys = cos_sig.evaluate(ts)
    
    wave = thinkdsp.Wave(ys, framerate=N)

    if window:
        wave.hamming()

    dct = wave.make_spectrum()
    dct.plot()


def main():
    N = 64
    
    #test_dct(N, freq=N/4)
    #test_dct(N, freq=N/4 + 0.25)

    test_fft(N, freq=N/4 + 0.25, window=True)
    test_dct(N, freq=N/4 + 0.25, window=True)

    thinkplot.show()
    return

    return


if __name__ == '__main__':
    main()
