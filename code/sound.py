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

class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class WavFile(object):

    def __init__(self, filename='sound.wav', framerate=11025):
        """Initializes the wave.
        """
        self.framerate = framerate
        self.nchannels = 1
        self.sampwidth = 2
        self.bits = self.sampwidth * 8
        self.bound = 2**(self.bits-1) - 1
        self.fmt = 'h'
        self.dtype = numpy.int16

        self.fp = wave.open(filename, 'w')
        self.fp.setnchannels(self.nchannels)
        self.fp.setsampwidth(self.sampwidth)
        self.fp.setframerate(self.framerate)
    
    def write(self, wave):
        """Writes the wave.

        wave: Wave
        """
        zs = wave.quantize(self.bound, self.dtype)
        oframes = array.array(self.fmt, zs)
        self.fp.writeframes(oframes)

    def close(self, duration=3):
        """Closes the file.

        duration: how many seconds of silence to append
        """
        if duration:
            self.write(rest(duration))

        self.fp.close()


class Wave(object):
    """Represents a discrete-time waveform."""

    def __init__(self, ys, framerate):
        """Initializes the wave.
        """
        self.ys = ys
        self.framerate = framerate

    def __or__(self, other):
        """
        """
        if self.framerate != other.framerate:
            raise ValueError('Wave.__or__: framerates do not agree')

        ys = numpy.concatenate((self.ys, other.ys))
        return Wave(ys, self.framerate)

    def quantize(self, bound, dtype):
        """Maps the waveform to quanta.

        ys: signal array
        bound: maximum amplitude
        dtype: numpy data type or string

        returns: quantized signal
        """
        return quantize(self.ys, bound, dtype)

    def apodize(self, denom=20, duration=0.1):
        """Tapers the amplitude at the beginning and end of the signal.

        Tapers either the given duration of time or the given
        fraction of the total duration, whichever is less.

        denom: float fraction of the sample to taper
        duration: float duration of the taper in seconds

        returns: signal array
        """
        self.ys = apodize(self.ys, self.framerate, denom, duration)

    def plot(self):
        """Plots the signal.

        """
        pyplot.plot(self.ts, self.ys)
        pyplot.show()


def rest(duration):
    signal = SilentSignal()
    wave = signal.make_wave(duration)
    return wave


def note(midi_num, duration):
    freq = midi_to_freq(midi_num)
    signal = CosSignal(freq)
    wave = signal.make_wave(duration)
    wave.apodize()
    return wave


def chord(midi_nums, duration):
    freqs = [midi_to_freq(num) for num in midi_nums]
    signal = sum(CosSignal(freq) for freq in freqs)
    wave = signal.make_wave(duration)
    wave.apodize()
    return wave


def midi_to_freq(midi_num):
    x = (midi_num - 69) / 12.0
    freq = 440.0 * 2**x
    return freq


def unbias(ys):
    """
    """
    return ys - ys.mean()


def normalize(ys):
    """
    """
    high, low = abs(max(ys)), abs(min(ys))
    return ys / max(high, low)


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: signal array
    bound: maximum amplitude
    dtype:

    returns: quantized signal
    """
    if max(ys) > 1 or min(ys) < -1:
        print 'Warning: normalizing before quantizing.'
        ys = normalize(ys)
        
    zs = (ys * bound).astype(dtype)
    return zs

def apodize(ys, framerate, denom=20, duration=0.1):
    """Tapers the amplitude at the beginning and end of the signal.

    Tapers either the given duration of time or the given
    fraction of the total duration, whichever is less.

    ys: signal array
    framerate: int frames per second
    denom: float fraction of the sample to taper
    duration: float duration of the taper in seconds

    returns: signal array
    """
    # a fixed fraction of the sample
    n = len(ys)
    k1 = n / denom

    # a fixed duration of time
    k2 = int(duration * framerate)

    k = min(k1, k2)

    w1 = numpy.linspace(0, 1, k)
    w2 = numpy.ones(n - 2*k)
    w3 = numpy.linspace(1, 0, k)

    window = numpy.concatenate((w1, w2, w3))
    return ys * window


class Signal(object):
    """Represents a time-varying signal."""

    def __add__(self, other):
        if other == 0:
            return self
        return SumSignal(self, other)

    __radd__ = __add__

    def make_wave(self, duration=1, start=0, framerate=11025):
        dt = 1.0 / framerate
        ts = numpy.arange(start, duration, dt)
        ys = self.evaluate(ts)
        return Wave(ys, framerate)


class SumSignal(Signal):
    """Represents the sum of signals."""
    
    def __init__(self, *args):
        """Initializes the sum.

        args: tuple of signals
        """
        self.signals = args

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float signal array
        """
        return sum(sig.evaluate(ts) for sig in self.signals)


class CosSignal(Signal):
    """Represents a cosine signal."""
    
    def __init__(self, freq=440, amp=1.0, offset=0):
        """Initializes a cosine signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        offset: float phase offset in radians
        """
        Signal.__init__(self)
        self.freq = freq
        self.amp = amp
        self.offset = offset

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float signal array
        """
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * numpy.cos(phases)
        return ys


class SilentSignal(Signal):
    """Represents silence."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float signal array
        """
        return numpy.zeros(len(ts))


def sin_signal(ts, freq=440, amp=1.0, offset=0):
    return amp * numpy.sin(PI2 * freq * ts + offset)


def func_signal(ts, func, freq=440, amp=1.0, offset=0):
    phase = freq * ts + offset
    phase, _ = numpy.modf(freq * ts)
    phase *= PI2

    ys = amp * func(phase)
    return ys

    return ys * window


def main():
    wfile = WavFile()
    for m in range(20, 120, 5):
        wfile.write(note(m, 0.25))
    wfile.close()
    return

    wave1 = note(69, 1)
    wave2 = chord([69, 72, 76], 1)
    wave = wave1 | wave2

    wfile = WavFile()
    wfile.write(wave)
    wfile.close()
    return

    sig1 = CosSignal(freq=440)
    sig2 = CosSignal(freq=523.25)
    sig3 = CosSignal(freq=660)
    sig4 = CosSignal(freq=880)
    sig5 = CosSignal(freq=987)
    sig = sig1 + sig2 + sig3 + sig4

    #wave = Wave(sig, duration=0.02)
    #wave.plot()

    wave = sig.make_wave(duration=1)
    #wave.normalize()

    wfile = WavFile(wave)
    wfile.write()



if __name__ == '__main__':
    main()
