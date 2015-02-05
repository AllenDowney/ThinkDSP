"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import array
import copy
import math
import numpy
import random
import scipy
import scipy.stats
import scipy.fftpack
import struct
import subprocess
import thinkplot
import warnings

from fractions import gcd
from wave import open as open_wave

import matplotlib.pyplot as pyplot

try:
    from IPython.display import Audio
except:
    warnings.warn("Can't import Audio from IPython.display; "
                  "Wave.make_audio() will not work.")

PI2 = math.pi * 2


def random_seed(x):
    """Initialize the random and numpy.random generators.

    x: int seed
    """
    random.seed(x)
    numpy.random.seed(x)


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

    dtype_map = {1:numpy.int8, 2:numpy.int16, 3:'special', 4:numpy.int32}
    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' % sampwidth)
    
    if sampwidth == 3:
        xs = numpy.fromstring(z_str, dtype=numpy.int8).astype(numpy.int32)
        ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
    else:
        ys = numpy.fromstring(z_str, dtype=dtype_map[sampwidth])

    # if it's in stereo, just pull out the first channel
    if nchannels == 2:
        ys = ys[::2]

    wave = Wave(ys, framerate)
    return wave


def play_wave(filename='sound.wav', player='aplay'):
    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    cmd = '%s %s' % (player, filename)
    popen = subprocess.Popen(cmd, shell=True)
    popen.communicate()


class _SpectrumParent(object):
    """Contains code common to Spectrum and DCT.
    """

    def copy(self):
        """Makes a copy.

        Returns: new Spectrum
        """
        return copy.deepcopy(self)

    @property
    def max_freq(self):
        return self.framerate / 2.0
        
    @property
    def freq_res(self):
        return self.max_freq / (len(self.fs) - 1)

    def plot(self, low=0, high=None, **options):
        """Plots amplitude vs frequency.

        low: int index to start at 
        high: int index to end at
        """
        thinkplot.plot(self.fs[low:high], self.amps[low:high], **options)

    def plot_power(self, low=0, high=None, **options):
        """Plots power vs frequency.

        low: int index to start at 
        high: int index to end at
        """
        thinkplot.plot(self.fs[low:high], self.power[low:high], **options)

    def estimate_slope(self):
        """Runs linear regression on log power vs log frequency.

        returns: slope, inter, r2, p, stderr
        """
        x = numpy.log(self.fs[1:])
        y = numpy.log(self.power[1:])
        t = scipy.stats.linregress(x,y)
        return t

    def peaks(self):
        """Finds the highest peaks and their frequencies.

        returns: sorted list of (amplitude, frequency) pairs
        """
        t = zip(self.amps, self.fs)
        t.sort(reverse=True)
        return t


class Spectrum(_SpectrumParent):
    """Represents the spectrum of a signal."""

    def __init__(self, hs, framerate):
        """Initializes a spectrum.

        hs: NumPy array of complex
        framerate: frames per second
        """
        self.hs = hs
        self.framerate = framerate

        # the frequency for each component of the spectrum depends
        # on whether the length of the wave is even or odd.
        # see http://docs.scipy.org/doc/numpy/reference/generated/
        # numpy.fft.rfft.html
        n = len(hs)
        if n%2 == 0:
            max_freq = self.max_freq
        else:
            max_freq = self.max_freq * (n-1) / n
            
        self.fs = numpy.linspace(0, max_freq, n)

    def __len__(self):
        """Length of the spectrum."""
        return len(self.hs)

    def __add__(self, other):
        """Adds two spectrums elementwise.

        other: Spectrum

        returns: new Spectrum
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate
        hs = self.hs + other.hs
        return Spectrum(hs, self.framerate)

    __radd__ = __add__
        
    def __mul__(self, other):
        """Multiplies two spectrums.

        other: Spectrum

        returns: new Spectrum
        """
        # the spectrums have to have the same framerate and duration
        assert self.framerate == other.framerate
        assert len(self) == len(other)

        hs = self.hs * other.hs
        return Spectrum(hs, self.framerate)
        
    @property
    def real(self):
        """Returns the real part of the hs (read-only property)."""
        return numpy.real(self.hs)

    @property
    def imag(self):
        """Returns the imaginary part of the hs (read-only property)."""
        return numpy.imag(self.hs)

    @property
    def amps(self):
        """Returns a sequence of amplitudes (read-only property)."""
        return numpy.absolute(self.hs)

    @property
    def power(self):
        """Returns a sequence of powers (read-only property)."""
        return self.amps ** 2

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
                self.hs[i] *= factor

    def pink_filter(self, beta=1):
        """Apply a filter that would make white noise pink.

        beta: exponent of the pink noise
        """
        denom = self.fs ** (beta/2.0)
        denom[0] = 1
        self.hs /= denom

    def differentiate(self):
        """Apply the differentiation filter.
        """
        i = complex(0, 1)
        filtr = PI2 * i * self.fs
        self.hs *= filtr

    def angles(self):
        """Computes phase angles in radians.

        returns: list of phase angles
        """
        return numpy.angle(self.hs)

    def make_integrated_spectrum(self):
        """Makes an integrated spectrum.
        """
        cs = numpy.cumsum(self.power)
        cs /= cs[-1]
        return IntegratedSpectrum(cs, self.fs)

    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        ys = numpy.fft.irfft(self.hs)
        return Wave(ys, self.framerate)


class IntegratedSpectrum(object):
    """Represents the integral of a spectrum."""
    
    def __init__(self, cs, fs):
        """Initializes an integrated spectrum:

        cs: sequence of cumulative amplitudes
        fs: sequence of frequences
        """
        self.cs = cs
        self.fs = fs

    def plot_power(self, low=0, high=None, expo=False, **options):
        """Plots the integrated spectrum.

        low: int index to start at 
        high: int index to end at
        """
        cs = self.cs[low:high]
        fs = self.fs[low:high]

        if expo:
            cs = numpy.exp(cs)

        thinkplot.plot(fs, cs, **options)

    def estimate_slope(self, low=1, high=-12000):
        """Runs linear regression on log cumulative power vs log frequency.

        returns: slope, inter, r2, p, stderr
        """
        #print self.fs[low:high]
        #print self.cs[low:high]
        x = numpy.log(self.fs[low:high])
        y = numpy.log(self.cs[low:high])
        t = scipy.stats.linregress(x,y)
        return t


class Dct(_SpectrumParent):
    """Represents the spectrum of a signal using discrete cosine transform."""

    def __init__(self, amps, framerate):
        self.amps = amps
        self.framerate = framerate
        n = len(amps)
        self.fs = numpy.arange(n) / float(n) * self.max_freq

    def __add__(self, other):
        """Adds two DCTs elementwise.

        other: DCT

        returns: new DCT
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate
        amps = self.amps + other.amps
        return Dct(amps, self.framerate)

    __radd__ = __add__
        
    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        ys = scipy.fftpack.dct(self.amps, type=3) / 2
        return Wave(ys, self.framerate)


class Spectrogram(object):
    """Represents the spectrum of a signal."""

    def __init__(self, spec_map, seg_length=512, window_func=None):
        """Initialize the spectrogram.

        spec_map: map from float time to Spectrum
        seg_length: number of samples in each segment
        window_func: function that computes the window
        """
        self.spec_map = spec_map
        self.seg_length = seg_length
        self.window_func = window_func

    def any_spectrum(self):
        """Returns an arbitrary spectrum from the spectrogram."""
        return self.spec_map.itervalues().next()

    @property
    def time_res(self):
        """Time resolution in seconds."""
        spectrum = self.any_spectrum()
        return float(self.seg_length) / spectrum.framerate

    @property
    def freq_res(self):
        """Frequency resolution in Hz."""
        return self.any_spectrum().freq_res

    def times(self):
        """Sorted sequence of times.

        returns: sequence of float times in seconds
        """
        ts = sorted(self.spec_map.iterkeys())
        return ts

    def frequencies(self):
        """Sequence of frequencies.

        returns: sequence of float freqencies in Hz.
        """
        fs = self.any_spectrum().fs
        return fs

    def plot(self, low=0, high=None, **options):
        """Make a pseudocolor plot.

        low: index of the lowest frequency component to plot
        high: index of the highest frequency component to plot
        """
        ts = self.times()
        fs = self.frequencies()[low:high]

        # make the array
        size = len(fs), len(ts)
        array = numpy.zeros(size, dtype=numpy.float)

        # copy amplitude from each spectrum into a column of the array
        for i, t in enumerate(ts):
            spectrum = self.spec_map[t]
            array[:,i] = spectrum.amps[low:high]

        thinkplot.pcolor(ts, fs, array, **options)

    def make_wave(self):
        """Inverts the spectrogram and returns a Wave.

        returns: Wave
        """
        res = []
        for t, spectrum in sorted(self.spec_map.iteritems()):
            wave = spectrum.make_wave()
            n = len(wave)
            
            if self.window_func:
                window = 1 / self.window_func(n)
                wave.window(window)

            i = int(round(t * wave.framerate))
            start = i - n / 2
            end = start + n
            res.append((start, end, wave))

        starts, ends, waves = zip(*res)
        low = min(starts)
        high = max(ends)

        ys = numpy.zeros(high-low, numpy.float)
        for start, end, wave in res:
            ys[start:end] = wave.ys

        return Wave(ys, wave.framerate)


class Wave(object):
    """Represents a discrete-time waveform.

    Note: the ys attribute is a "wave array" which is a numpy
    array of floats.
    """

    def __init__(self, ys, framerate, start=0):
        """Initializes the wave.

        ys: wave array
        framerate: samples per second
        """
        self.ys = ys
        self.framerate = framerate
        self.start = start

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.ys)

    @property
    def duration(self):
        """Duration (property).

        returns: float duration in seconds
        """
        return len(self.ys) / float(self.framerate)

    def __add__(self, other):
        """Adds two waves elementwise.

        other: Wave

        returns: new Wave
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate
        n1, n2 = len(self), len(other)
        if n1 > n2:
            ys = self.ys.copy()
            ys[:n2] += other.ys
        else:
            ys = other.ys.copy()
            ys[:n1] += self.ys
            
        return Wave(ys, self.framerate)

    __radd__ = __add__
        
    def __or__(self, other):
        """Concatenates two waves.

        other: Wave
        
        returns: Wave
        """
        if self.framerate != other.framerate:
            raise ValueError('Wave.__or__: framerates do not agree')

        ys = numpy.concatenate((self.ys, other.ys))
        return Wave(ys, self.framerate)

    def __mul__(self, other):
        """Convolves two waves.

        other: Wave
        
        returns: Wave
        """
        if self.framerate != other.framerate:
            raise ValueError('Wave convolution: framerates do not agree')

        ys = numpy.convolve(self.ys, other.ys, mode='full')
        ys = ys[:len(self.ys)]
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

    def hamming(self):
        """Apply a Hamming window to the wave.
        """
        self.ys *= numpy.hamming(len(self.ys))

    def window(self, window):
        """Apply a window to the wave.

        window: sequence of multipliers, same length as self.ys
        """
        self.ys *= window

    def scale(self, factor):
        """Multplies the wave by a factor.

        factor: scale factor
        """
        self.ys *= factor

    def shift(self, shift):
        """Shifts the wave left or right by index shift.

        shift: integer number of places to shift
        """
        if shift < 0:
            self.ys = shift_left(self.ys, shift)
        if shift > 0:
            self.ys = shift_right(self.ys, shift)
        
    def truncate(self, n):
        """Trims this wave to the given length.
        """
        self.ys = truncate(self.ys, n)

    def normalize(self, amp=1.0):
        """Normalizes the signal to the given amplitude.

        amp: float amplitude
        """
        self.ys = normalize(self.ys, amp=amp)

    def unbias(self):
        """Unbiases the signal.
        """
        self.ys = unbias(self.ys)

    def segment(self, start=0, duration=None):
        """Extracts a segment.

        start: float start time in seconds
        duration: float duration in seconds

        returns: Wave
        """
        i = round(start * self.framerate)

        if duration is None:
            j = None
        else:
            j = i + round(duration * self.framerate)

        ys = self.ys[i:j]
        return Wave(ys, self.framerate)

    def make_spectrum(self):
        """Computes the spectrum using FFT.

        returns: Spectrum
        """
        hs = numpy.fft.rfft(self.ys)
        return Spectrum(hs, self.framerate)

    def make_dct(self):
        amps = scipy.fftpack.dct(self.ys, type=2)
        return Dct(amps, self.framerate)

    def make_spectrogram(self, seg_length, window_func=numpy.hamming):
        """Computes the spectrogram of the wave.

        seg_length: number of samples in each segment
        window_func: function used to compute the window

        returns: Spectrogram
        """
        n = len(self.ys)
        window = window_func(seg_length)

        start, end, step = 0, seg_length, seg_length / 2
        spec_map = {}

        while end < n:
            ys = self.ys[start:end] * window
            hs = numpy.fft.rfft(ys)

            t = (start + end) / 2.0 / self.framerate
            spec_map[t] = Spectrum(hs, self.framerate)

            start += step
            end += step

        return Spectrogram(spec_map, seg_length, window_func)

    def plot(self, **options):
        """Plots the wave.

        """
        n = len(self.ys)
        ts = numpy.linspace(0, self.duration, n)
        thinkplot.plot(ts, self.ys, **options)

    def corr(self, other):
        """Correlation coefficient two waves.

        other: Wave

        returns: float coefficient of correlation
        """
        corr = numpy.corrcoef(self.ys, other.ys)[0, 1]
        return corr
        
    def cov_mat(self, other):
        """Covariance matrix of two waves.

        other: Wave

        returns: 2x2 covariance matrix
        """
        return numpy.cov(self.ys, other.ys)

    def cov(self, other):
        """Covariance of two unbiased waves.

        other: Wave

        returns: float
        """
        total = sum(self.ys * other.ys) / len(self.ys)
        return total

    def cos_cov(self, k):
        """Covariance with a cosine signal.

        freq: freq of the cosine signal in Hz

        returns: float covariance
        """
        n = len(self.ys)
        factor = math.pi * k / n
        ys = [math.cos(factor * (i+0.5)) for i in range(n)]
        total = 2 * sum(self.ys * ys)
        return total

    def cos_transform(self):
        """Discrete cosine transform.

        returns: list of frequency, cov pairs
        """
        n = len(self.ys)
        res = []
        for k in range(n):
            cov = self.cos_cov(k)
            res.append((k, cov))

        return res

    def write(self, filename='sound.wav'):
        """Write a wave file.

        filename: string
        """
        print('Writing', filename)
        wfile = WavFileWriter(filename, self.framerate)
        wfile.write(self)
        wfile.close()

    def play(self, filename='sound.wav'):
        """Plays a wave file.

        filename: string
        """
        self.write(filename)
        play_wave(filename)

    def make_audio(self):
        """Makes an IPython Audio object.
        """
        audio = Audio(data=self.ys, rate=self.framerate)
        return audio


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


def shift_right(ys, shift):
    """Shifts a wave array to the right and zero pads.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    res = numpy.zeros(len(ys) + shift)
    res[shift:] = ys
    return res


def shift_left(ys, shift):
    """Shifts a wave array to the left.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    return ys[shift:]


def truncate(ys, n):
    """Trims a wave array to the given length.

    ys: wave array
    n: integer length

    returns: wave array
    """
    return ys[:n]


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

    returns: quantized signal
    """
    if max(ys) > 1 or min(ys) < -1:
        warnings.warn('Warning: normalizing before quantizing.')
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
    k1 = n // denom

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
        """Period of the signal in seconds (property).

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
        return Wave(ys, framerate=framerate, start=start)


def infer_framerate(ts):
    """Given ts, find the framerate.

    Assumes that the ts are equally spaced.

    ts: sequence of times in seconds

    returns: frames per second
    """
    dt = ts[1] - ts[0]
    framerate = 1.0 / dt
    return framerate


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
    """Makes a cosine Sinusoid.

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


class ComplexSinusoid(Sinusoid):
    """Represents a complex exponential signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        i = complex(0, 1)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * numpy.exp(i * phases)
        return ys


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
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = numpy.modf(cycles)
        ys = numpy.abs(frac - 0.5)
        ys = normalize(unbias(ys), self.amp)
        return ys


class Chirp(Signal):
    """Represents a signal with variable frequency."""
    
    def __init__(self, start=440, end=880, amp=1.0):
        """Initializes a linear chirp.

        start: float frequency in Hz
        end: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        """
        self.start = start
        self.end = end
        self.amp = amp

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return ValueError('Non-periodic signal.')

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        freqs = numpy.linspace(self.start, self.end, len(ts)-1)
        return self._evaluate(ts, freqs)

    def _evaluate(self, ts, freqs):
        """Helper function that evaluates the signal.

        ts: float array of times
        freqs: float array of frequencies during each interval
        """
        dts = numpy.diff(ts)
        dps = PI2 * freqs * dts
        phases = numpy.cumsum(dps)
        phases = numpy.insert(phases, 0, 0)
        ys = self.amp * numpy.cos(phases)
        return ys


class ExpoChirp(Chirp):
    """Represents a signal with varying frequency."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        start, end = math.log10(self.start), math.log10(self.end)
        freqs = numpy.logspace(start, end, len(ts)-1)
        return self._evaluate(ts, freqs)


class SilentSignal(Signal):
    """Represents silence."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        return numpy.zeros(len(ts))


class _Noise(Signal):
    """Represents a noise signal (abstract parent class)."""
    
    def __init__(self, amp=1.0):
        """Initializes a white noise signal.

        amp: float amplitude, 1.0 is nominal max
        """
        self.amp = amp

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return ValueError('Non-periodic signal.')


class UncorrelatedUniformNoise(_Noise):
    """Represents uncorrelated uniform noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        ys = numpy.random.uniform(-self.amp, self.amp, len(ts))
        return ys


class UncorrelatedGaussianNoise(_Noise):
    """Represents uncorrelated gaussian noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        ys = numpy.random.normal(0, self.amp, len(ts))
        return ys


class BrownianNoise(_Noise):
    """Represents Brownian noise, aka red noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        Computes Brownian noise by taking the cumulative sum of
        a uniform random series.

        ts: float array of times
        
        returns: float wave array
        """
        dys = numpy.random.uniform(-1, 1, len(ts))
        #ys = scipy.integrate.cumtrapz(dys, ts)
        ys = numpy.cumsum(dys)
        ys = normalize(unbias(ys), self.amp)
        return ys


class PinkNoise(_Noise):
    """Represents Brownian noise, aka red noise."""

    def __init__(self, amp=1.0, beta=1.0):
        """Initializes a pink noise signal.

        amp: float amplitude, 1.0 is nominal max
        """
        self.amp = amp
        self.beta = beta

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        signal = UncorrelatedUniformNoise()
        wave = signal.make_wave(duration, start, framerate)
        spectrum = wave.make_spectrum()

        spectrum.pink_filter(beta=self.beta)

        wave2 = spectrum.make_wave()
        wave2.unbias()
        wave2.normalize(self.amp)
        return wave2


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

    wave = cos_wave(440, offset=math.pi/2)
    cos_cov = cos_basis.cov(wave)
    sin_cov = sin_basis.cov(wave)
    print(cos_cov, sin_cov, mag((cos_cov, sin_cov)))
    return

    wfile = WavFileWriter()
    for sig_cons in [SinSignal, TriangleSignal, SawtoothSignal, 
                     GlottalSignal, ParabolicSignal, SquareSignal]:
        print(sig_cons)
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
