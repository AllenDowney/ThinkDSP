"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import numpy
import thinkdsp
import thinkplot

PI2 = 2 * math.pi

class SawtoothChirp(thinkdsp.Chirp):
    """Represents a sawtooth signal with varying frequency."""

    def _evaluate(self, ts, freqs):
        """Helper function that evaluates the signal.

        ts: float array of times
        freqs: float array of frequencies during each interval
        """
        dts = numpy.diff(ts)
        dps = PI2 * freqs * dts
        phases = numpy.cumsum(dps)
        phases = numpy.insert(phases, 0, 0)
        cycles = phases / PI2
        frac, _ = numpy.modf(cycles)
        ys = thinkdsp.normalize(thinkdsp.unbias(frac), self.amp)
        return ys


class TromboneGliss(thinkdsp.Chirp):
    """Represents a trombone-like signal with varying frequency."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        l1, l2 = 1.0 / self.start, 1.0 / self.end
        lengths = numpy.linspace(l1, l2, len(ts)-1)
        freqs = 1 / lengths
        return self._evaluate(ts, freqs)


def sawtooth_chirp():
    """Tests SawtoothChirp.
    """
    signal = SawtoothChirp(start=220, end=880)
    wave = signal.make_wave(duration=2, framerate=44100)
    wave.apodize()
    wave.play()

    sp = wave.make_spectrogram(1024)
    sp.plot()
    thinkplot.show()


def trombone_gliss():
    """Tests TromboneGliss.
    """
    low = 262
    high = 340
    signal = TromboneGliss(high, low)
    wave1 = signal.make_wave(duration=1)
    wave1.apodize()

    signal = TromboneGliss(low, high)
    wave2 = signal.make_wave(duration=1)
    wave2.apodize()

    wave = wave1 | wave2
    filename = 'gliss.wav'
    wave.write(filename)
    thinkdsp.play_wave(filename)

    sp = wave.make_spectrogram(1024)
    sp.plot(high=40)
    thinkplot.show()


def main():
    sawtooth_chirp()
    trombone_gliss()
    return




if __name__ == '__main__':
    main()
