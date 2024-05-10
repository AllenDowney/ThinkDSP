# Упражнение 2.2

from thinkdsp import Sinusoid, TriangleSignal
from thinkdsp import normalize, unbias, decorate
import numpy as np


class SawtoothSignal(Sinusoid):
    """Sawtooth signal."""

    def evaluate(self, ts):
        cycles = self.freq * ts + self.offset / np.pi / 2
        frac, _ = np.modf(cycles)
        ys = normalize(unbias(frac), self.amp)
        return ys


sawtooth = SawtoothSignal().make_wave(duration=0.5, framerate=40000)
sawtooth.make_audio()

sawtooth.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')

sawtooth.make_spectrum().plot(color='gray')
triangle = TriangleSignal(amp=0.79).make_wave(duration=0.5, framerate=40000)
triangle.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')
