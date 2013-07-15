"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import numpy
import wave

PI2 = math.pi * 2

def write_wav(ys, filename='sound.wav', sample_rate=11025):
    fp = wave.open(filename, 'w')
    fp.setsampwidth(2)
    fp.setnchannels(1)
    fp.setframerate(sample_rate)
    fp.setnframes(len(ys))
    fp.writeframes(ys)
    fp.close()


def discretize(ys, bits=16):
    bound = 2**(bits-1) - 1
    zs = (ys * bound)
    return zs.astype('int16')


def sample_times(end=1, sample_rate=11025):
    dt = 1.0 / sample_rate
    return numpy.arange(0, end, dt)


def cos_signal(ts, freq=440, amp=1.0, phase=0):
    return amp * numpy.cos(PI2 * freq * ts + phase)


def apodize(ys):
    window = numpy.kaiser(len(ys), beta=5)
    return window * ys
    

def main():
    ts = sample_times(2)
    print ts[0], ts[-1], len(ts)

    ys = cos_signal(ts, freq=440)
    print ys[0], ys[-1], len(ys)

    ys = apodize(ys)

    zs = discretize(ys)
    print zs[0], zs[-1], len(zs), max(zs), min(zs)

    write_wav(zs)

if __name__ == '__main__':
    main()
