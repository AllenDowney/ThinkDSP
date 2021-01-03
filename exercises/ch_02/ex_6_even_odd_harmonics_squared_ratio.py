 # Triangle and square waves have odd harmonics only; the sawtooth wave has both even and
# odd harmonics. The harmonics of the square and sawtooth waves drop off in proportion to 1/f
# the harmonics of the triangle wave drop off like 1/f^2.
# Can you find a waveform that has even and odd harmonics that drop off like 1/f^2? 

import numpy as np

from code.thinkdsp import PI2, Sinusoid, TriangleSignal, decorate, normalize, unbias
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22500


class TriangleShiftSignal(Sinusoid):
    def evaluate(self, ts):
        cycles = self.freq * ts + self.offset / PI2
        shifted_cycles = self.freq * ts + (self.offset + 0.5) / PI2
        frac, _ = np.modf(cycles + shifted_cycles)
        ys = np.abs(frac - 0.5)
        ys = normalize(unbias(ys), self.amp)
        return ys

def run():
    # Unshifted triangle, this should have harmonics only at odd multiples of the fundamental
    sig = TriangleSignal(freq=FREQ_A4, amp=0.5, offset=0)

    start = 0.
    duration_secs = 1.
    segment = sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)

    spectrum = segment.make_spectrum()
    print('Plotting triangle signal spectrum')
    wave = spectrum.make_wave()
    wave.plot()
    decorate(xlabel='Time (s)')

    print('Playing triangle signal spectrum')
    play_wave(wave)

    # Shifted triangle, standard triangle plus a second triangle phased-shifted by half a period
    #  to include additional harmonic content
    sig = TriangleShiftSignal(freq=FREQ_A4, amp=0.5, offset=0)
    segment = sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)

    spectrum = segment.make_spectrum()
    print('Plotting shifted triangle signal spectrum')
    wave = spectrum.make_wave()
    wave.plot()
    decorate(xlabel='Time (s)')

    print('Playing shifted triangle signal spectrum')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 2: ex_5_modify_spectrum_by_frequency.py")
    print("****************************")
    run()
