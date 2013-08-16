"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import array
import math
import numpy
import struct
import subprocess
import thinkplot

from fractions import gcd
from wave import open as open_wave

import matplotlib.pyplot as pyplot

PI2 = math.pi * 2


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class WavFileWriter(object):
    """Writes wav files."""

    def __init__(self, filename='sound.wav', framerate=11025):
        """Opens the file and sets parameters.

        filename: string
        framerate: samples per second
        """
        self.filename = filename
        self.framerate = framerate
        self.nchannels = 1
        self.sampwidth = 2
        self.bits = self.sampwidth * 8
        self.bound = 2**(self.bits-1) - 1

        self.fmt = 'h'
        self.dtype = numpy.int16

        self.fp = open_wave(self.filename, 'w')
        self.fp.setnchannels(self.nchannels)
        self.fp.setsampwidth(self.sampwidth)
        self.fp.setframerate(self.framerate)
    
    def write(self, wave):
        """Writes a wave.

        wave: Wave
        """
        zs = wave.quantize(self.bound, self.dtype)
        self.fp.writeframes(zs.tostring())

    def close(self, duration=0):
        """Closes the file.

        duration: how many seconds of silence to append
        """
        if duration:
            self.write(rest(duration))

        self.fp.close()


def read_wave(filename='sound.wav'):
    """Reads a wave file.

    filename: string

    returns: Wave
    """
    fp = open_wave(filename, 'r')

    nchannels = fp.getnchannels()
    nframes = fp.getnframes()
    sampwidth = fp.getsampwidth()
    framerate = fp.getframerate()
    
    z_str = fp.readframes(nframes)
    
    fp.close()

    # TODO: generalize this to handle other sample widths
    assert sampwidth == 2
    fmt = 'h'
    dtype = numpy.int16

    ys = numpy.fromstring(z_str, dtype=dtype)
    wave = Wave(ys, framerate)
    return wave


def play_wave(filename='sound.wav', player='aplay'):
    """Playes a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    cmd = '%s %s' % (player, filename)
    subprocess.Popen(cmd, shell=True)

    # TODO: join?


class Spectrum(object):
    """Represents the spectrum of a signal."""

    def __init__(self, hs, framerate):
        self.hs = hs
        self.framerate = framerate

        self.amps = numpy.absolute(self.hs)

        n = len(hs)
        f_max = framerate / 2.0
        self.fs = numpy.linspace(0, f_max, n)

    def plot(self, low=0, high=None):
        """Plots the spectrum.

        low: int index to start at 
        high: int index to end at
        """
        thinkplot.Plot(self.fs[low:high], self.amps[low:high])

    def peaks(self):
        """Finds the highest peaks and their frequencies.

        returns: sorted list of (amplitude, frequency) pairs
        """
        t = zip(self.amps, self.fs)
        t.sort(reverse=True)
        return t

    def low_pass(self, cutoff, factor=0):
        """Attenuate frequencies above the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        for i in xrange(len(self.hs)):
            if self.fs[i] > cutoff:
                self.hs[i] *= factor

    def high_pass(self, cutoff, factor=0):
        """Attenuate frequencies below the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        for i in xrange(len(self.hs)):
            if self.fs[i] < cutoff:
                self.hs[i] *= factor

    def band_stop(self, low_cutoff, high_cutoff, factor=0):
        """Attenuate frequencies between the cutoffs.

        low_cutoff: frequency in Hz
        high_cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        for i in xrange(len(self.hs)):
            if low_cutoff < self.fs[i] < high_cutoff:
                self.hs[i] = 0

    def angles(self, i):
        """Computes phase angles in radians.

        returns: list of phase angles
        """
        return numpy.angle(self.hs)

    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        ys = numpy.fft.irfft(self.hs)
        return Wave(ys, self.framerate)


class Wave(object):
    """Represents a discrete-time waveform.

    Note: the ys attribute is a "wave array" which is a numpy
    array of floats.
    """

    def __init__(self, ys, framerate):
        """Initializes the wave.

        ys: wave array
        framerate: samples per second
        """
        self.ys = ys
        self.framerate = framerate

    def __or__(self, other):
        """Concatenates two waves.

        other: Wave
        
        returns: Wave
        """
        if self.framerate != other.framerate:
            raise ValueError('Wave.__or__: framerates do not agree')

        ys = numpy.concatenate((self.ys, other.ys))
        return Wave(ys, self.framerate)

    def quantize(self, bound, dtype):
        """Maps the waveform to quanta.

        bound: maximum amplitude
        dtype: numpy data type or string

        returns: quantized signal
        """
        return quantize(self.ys, bound, dtype)

    def apodize(self, denom=20, duration=0.1):
        """Tapers the amplitude at the beginning and end of the signal.

        Tapers either the given duration of time or the given
        fraction of the total duration, whichever is less.

        denom: float fraction of the segment to taper
        duration: float duration of the taper in seconds
        """
        self.ys = apodize(self.ys, self.framerate, denom, duration)

    def normalize(self, amp=1.0):
        """Normalizes the signal to the given amplitude.

        amp: float amplitude

        returns: sequence of floats
        """
        self.ys = normalize(self.ys, amp=amp)

    def segment(self, start, duration):
        """Extracts a segment.

        start: float start time in seconds
        duration: float duration in seconds

        returns: Wave
        """
        i = start * self.framerate
        j = i + duration * self.framerate
        ys = self.ys[i:j]
        return Wave(ys, self.framerate)

    def make_spectrum(self):
        """Computes the spectrum using FFT.

        returns: Spectrum
        """
        hs = numpy.fft.rfft(self.ys)
        return Spectrum(hs, self.framerate)

    def plot(self):
        """Plots the wave.

        """
        n = len(self.ys)
        duration = float(n) / self.framerate
        ts = numpy.linspace(0, duration, n)
        thinkplot.Plot(ts, self.ys)

    def cov(self, other):
        """Computes the covariance of two waves.

        other: Wave

        returns: float
        """
        total = sum(self.ys * other.ys)
        return total / len(self.ys)

    def write(self, filename='sound.wav'):
        """Write a wave file.

        filename: string
        """
        print 'Writing', filename
        wfile = WavFileWriter(filename, self.framerate)
        wfile.write(self)
        wfile.close()



def unbias(ys):
    """Shifts a wave array so it has mean 0.

    ys: wave array

    returns: wave array
    """
    return ys - ys.mean()


def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.

    ys: wave array
    amp: max amplitude (pos or neg) in result

    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

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

    ys: wave array
    framerate: int frames per second
    denom: float fraction of the segment to taper
    duration: float duration of the taper in seconds

    returns: wave array
    """
    # a fixed fraction of the segment
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
        """Adds two signals.

        other: Signal

        returns: Signal
        """
        if other == 0:
            return self
        return SumSignal(self, other)

    __radd__ = __add__

    @property
    def period(self):
        """Period of the signal in seconds.

        For non-periodic signals, use the default, 0.1 seconds

        returns: float seconds
        """
        return 0.1

    def plot(self, framerate=11025):
        """Plots the signal.

        framerate: samples per second
        """
        duration = self.period * 3
        wave = self.make_wave(duration, start=0, framerate=framerate)
        wave.plot()
    
    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
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

    @property
    def period(self):
        """Period of the signal in seconds.

        Note: this is not correct; it's mostly a placekeeper.

        But it is correct for a harmonic sequence where all
        component frequencies are multiples of the fundamental.

        returns: float seconds
        """
        return max(sig.period for sig in self.signals)

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        return sum(sig.evaluate(ts) for sig in self.signals)


class Sinusoid(Signal):
    """Represents a sinusoidal signal."""
    
    def __init__(self, freq=440, amp=1.0, offset=0, func=numpy.sin):
        """Initializes a sinusoidal signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        offset: float phase offset in radians
        func: function that maps phase to amplitude
        """
        Signal.__init__(self)
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return 1.0 / self.freq

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys


def CosSignal(freq=440, amp=1.0, offset=0):
    """Makes a consine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians
    
    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=numpy.cos)


def SinSignal(freq=440, amp=1.0, offset=0):
    """Makes a sine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians
    
    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=numpy.sin)


class SquareSignal(Sinusoid):
    """Represents a square signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = numpy.modf(cycles)
        ys = self.amp * numpy.sign(unbias(frac))
        return ys


class SawtoothSignal(Sinusoid):
    """Represents a sawtooth signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = numpy.modf(cycles)
        ys = normalize(unbias(frac), self.amp)
        return ys


class ParabolicSignal(Sinusoid):
    """Represents a parabolic signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = numpy.modf(cycles)
        ys = frac**2
        ys = normalize(unbias(ys), self.amp)
        return ys


class GlottalSignal(Sinusoid):
    """Represents a periodic signal that resembles a glottal signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = numpy.modf(cycles)
        ys = frac**4 * (1-frac)
        ys = normalize(unbias(ys), self.amp)
        return ys


class TriangleSignal(Sinusoid):
    """Represents a triangle signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        cycles = self.freq * ts
        frac, _ = numpy.modf(cycles)
        ys = numpy.abs(frac - 0.5)
        ys = normalize(unbias(ys), self.amp)
        return ys


class SilentSignal(Signal):
    """Represents silence."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        return numpy.zeros(len(ts))


def rest(duration):
    """Makes a rest of the given duration.

    duration: float seconds

    returns: Wave
    """
    signal = SilentSignal()
    wave = signal.make_wave(duration)
    return wave


def make_note(midi_num, duration, sig_cons=CosSignal, framerate=11025):
    """Make a MIDI note with the given duration.

    midi_num: int MIDI note number
    duration: float seconds
    sig_cons: Signal constructor function
    framerate: int frames per second

    returns: Wave
    """
    freq = midi_to_freq(midi_num)
    signal = sig_cons(freq)
    wave = signal.make_wave(duration, framerate=framerate)
    wave.apodize()
    return wave


def make_chord(midi_nums, duration, sig_cons=CosSignal, framerate=11025):
    """Make a chord with the given duration.

    midi_nums: sequence of int MIDI note numbers
    duration: float seconds
    sig_cons: Signal constructor function
    framerate: int frames per second

    returns: Wave
    """
    freqs = [midi_to_freq(num) for num in midi_nums]
    signal = sum(sig_cons(freq) for freq in freqs)
    wave = signal.make_wave(duration, framerate=framerate)
    wave.apodize()
    return wave


def midi_to_freq(midi_num):
    """Converts MIDI note number to frequency.

    midi_num: int MIDI note number
    
    returns: float frequency in Hz
    """
    x = (midi_num - 69) / 12.0
    freq = 440.0 * 2**x
    return freq


def sin_wave(freq, duration=1, offset=0):
    """Makes a sine wave with the given parameters.

    freq: float cycles per second
    duration: float seconds
    offset: float radians

    returns: Wave
    """
    signal = SinSignal(freq, offset=offset)
    wave = signal.make_wave(duration)
    return wave


def cos_wave(freq, duration=1, offset=0):
    """Makes a cosine wave with the given parameters.

    freq: float cycles per second
    duration: float seconds
    offset: float radians

    returns: Wave
    """
    signal = CosSignal(freq, offset=offset)
    wave = signal.make_wave(duration)
    return wave


def mag(a):
    """Computes the magnitude of a numpy array.

    a: numpy array

    returns: float
    """
    return numpy.sqrt(numpy.dot(a, a))


def main():

    cos_basis = cos_wave(440)
    sin_basis = sin_wave(440)

    wave = cos_wave(440, offset=1)
    cos_cov = cos_basis.cov(wave)
    sin_cov = sin_basis.cov(wave)
    print cos_cov, sin_cov, mag((cos_cov, sin_cov))
    return

    wfile = WavFileWriter()
    for sig_cons in [SinSignal, TriangleSignal, SawtoothSignal, 
                     GlottalSignal, ParabolicSignal, SquareSignal]:
        print sig_cons
        sig = sig_cons(440)
        wave = sig.make_wave(1)
        wave.apodize()
        wfile.write(wave)
    wfile.close()
    return

    signal = GlottalSignal(440)
    signal.plot()
    pyplot.show()
    return

    wfile = WavFileWriter()
    for m in range(60, 0, -1):
        wfile.write(make_note(m, 0.25))
    wfile.close()
    return

    wave1 = make_note(69, 1)
    wave2 = make_chord([69, 72, 76], 1)
    wave = wave1 | wave2

    wfile = WavFileWriter()
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

    wfile = WavFileWriter(wave)
    wfile.write()
    wfile.close()


if __name__ == '__main__':
    main()
