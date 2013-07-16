"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import array
import math
import numpy
import struct
import wave

import matplotlib.pyplot as pyplot

PI2 = math.pi * 2

def write_wav(ys, filename='sound.wav', sample_rate=11025):
    nchannels = 1
    sampwidth = 2
    fmt = 'h'

    fp = wave.open(filename, 'w')
    fp.setnchannels(nchannels)
    fp.setsampwidth(sampwidth)
    fp.setframerate(sample_rate)
    fp.setnframes(len(ys))

    if max(ys) > 32767 or min(ys) < -32768:
        raise ValueError('Signal does not fit in 16 bits')
        
    oframes = array.array(fmt, ys)
    print oframes[:1000]

    fp.writeframes(oframes)
    fp.close()


def discretize(ys, bits=16):
    bound = 2**(bits-1) - 1
    zs = (ys * bound)
    return zs.astype('int16')


def sample_times(end=1, sample_rate=11025):
    dt = 1.0 / sample_rate
    return numpy.arange(0, end, dt)


def cos_signal(ts, freq=440, amp=1.0, offset=0):
    return amp * numpy.cos(PI2 * freq * ts + offset)


def sin_signal(ts, freq=440, amp=1.0, offset=0):
    return amp * numpy.sin(PI2 * freq * ts + offset)


def func_signal(ts, func, freq=440, amp=1.0, offset=0):
    phase = freq * ts + offset
    phase, _ = numpy.modf(freq * ts)
    phase *= PI2

    ys = amp * func(phase)
    return ys


def apodize(ys, sample_rate=11025):
    # a fixed fraction of the sample
    n = len(ys)
    k1 = n / 20

    # a fixed duration of time
    k2 = sample_rate / 10

    k = min(k1, k2)

    w1 = numpy.linspace(0, 1, k)
    w2 = numpy.ones(n - 2*k)
    w3 = numpy.linspace(1, 0, k)
    
    window = numpy.concatenate((w1, w2, w3))

    #pyplot.plot(window)
    #pyplot.plot(ys)
    #pyplot.plot(ys * window)
    #pyplot.show()
    
    return ys * window


def main():
    ts = sample_times(1)
    print ts[0], ts[-1], len(ts)

    def identity(ts): return ts / PI2

    func = identity
    func = numpy.cos

    ys = func_signal(ts, func, freq=440)
    print ys[0], ys[-1], len(ys)

    #pyplot.plot(ys[:100])
    #pyplot.show()

    ys = apodize(ys)

    zs = discretize(ys)
    print zs[0], zs[-1], len(zs), min(zs), max(zs)

    write_wav(zs)


if __name__ == '__main__':
    main()
