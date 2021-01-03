# A sawtooth signal has a waveform that ramps up linearly from –1 to 1, then drops to –1 and repeats
# Write a class called SawtoothSignal that extends Signal and provides evaluate to evaluate
#  a sawtooth signal.
# Compute the spectrum of a sawtooth wave. How does the harmonic structure compare to
#  triangle and square waves?

import numpy as np

from code.thinkdsp import PI2, Sinusoid, normalize, unbias
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22050


# NOTE: Copied from the book and annotated
class TriangleSignal(Sinusoid):
    def evaluate(self, ts):
        # Generic logic for all oscillator evaluate() implementations
        # The samples periodic signal function since start time
        # Signal Freq * duration + offset (in radians)
        cycles = self.freq * ts + self.offset / PI2
        # Like a mod, converts cycles into an in integer part and a float remainder part, i.e.
        #  for the current partially completed cycle what is the offset into that in radians
        # NOTE: frac cycles in the range [0.0 .. 1.0)
        frac, _ = np.modf(cycles)
        # Logic specific to TriangleSignal
        # Subrtract 0.5:  move frace to range -0.5 .. 0.5
        # abs(): map all negative values to 0, so freq now in range 0.0 .. 0.5
        # unbias(), thinkdsp.py#L1107, normalizes a vector of samples over it's mean. In signal
        #  processing the intuition is that unbiased samples do not deviate from that they
        #  measure, that is they have an error of 0 between the wave (samples) and the
        #  signal function the wave represents. TODO: Why does normalizing over the mean
        #  achieve this?
        # The book says that this "centers the samples around 0" so this seems like another
        #  shift similar to subtracing 0.5.
        # normalize() then multiples the samples by the amp for the oscillator
        ys = np.abs(frac - 0.5)
        ys = normalize(unbias(ys), self.amp)
        return ys


# # NOTE: Copied from the book and annotated
# class SquareSignal(Sinusoid):
#     def evaluate(self, ts):
#         cycles = self.freq * ts + self.offset / PI2
#         frac, _ = np.modf(cycles)
#         # Center samples around 0 and then call sign() to map them all to {+1 if x >= 0, else -1}
#         ys = self.amp * np.sign(unbias(frac))
#         return ys


class SawtoothSignal(Sinusoid):
    def evaluate(self, ts):
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        frac -= 0.5
        frac = unbias(frac)
        # Map negative offsets to -1, like the negative half of a square wave
        # Just unbias positive values, making the positive half of the square wave behave
        #  like the triangle wave
        # Use the resulting mapping of all offsets to scale amp for the signal instance
        ys = np.where(frac >= 0.0, normalize(frac, self.amp), -1.0 * self.amp)
        return ys


def run():
    start = 0.
    duration_secs = 1.
    sawtooth_sig = SawtoothSignal(freq=FREQ_A4, amp=1.0, offset=0)
    wave = sawtooth_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing sawtooth wave')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 2: ex_2_sawtooth.py")
    print("****************************")
    run()

