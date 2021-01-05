 # A trombone player can play a glissando by extending the trombone slide while
 #  blowing continuously. As the slide extends, the total length of the tube gets longer,
 #  and the resulting pitch is inversely proportional to length.
 # Assuming that the player moves the slide at a constant speed, how does frequency vary
 #  with time?
 #  Expected: Logarithmic slope rise and fall
 # Write a class called TromboneGliss that extends Chirp and provides evaluate.
 # Make a wave that simulates a trombone glissando from C3 up to F3 and back down to C3.
 # C3 is 262 Hz; F3 is 349 Hz.

from enum import Enum
from math import floor

import numpy as np

from code.thinkdsp import PI2, Sinusoid, normalize, unbias
from exercises.lib.lib import play_wave

FRAMERATE = 22500
START_FREQ = 262
END_FREQ = 349
OFFSET = 0
START = 0
DURATION_SECS = 2
AMP = 0.5


# Chirp implementation From thinkdsp
class TromboneGlissandoSignal(Sinusoid):
    def __init__(self, low_freq, high_freq, **options):
        super(TromboneGlissandoSignal, self).__init__(**options)
        self._low_freq = low_freq
        self._high_freq = high_freq


    def evaluate(self, ts):
        num_samples = floor(len(ts) / 2)
        # Build the rise half of the glissando, this essentially the same code as Chirp
        ys_rise = self._evaluate(self._low_freq, self._high_freq,
                                 ts[0 : num_samples], num_samples)
        # Copy the rise half of the wave, flip it to get the fall half of the wave,
        # then concatenate the rise and fall
        ys_fall = np.flip(np.copy(ys_rise))
        ys = np.append(ys_rise, ys_fall)
        return ys


    def _evaluate(self, start_freq, end_freq, ts, num_samples):
        # Sample frequencies at even intervals from start to end
        freqs = np.linspace(start_freq, end_freq, num_samples)
        # Get the intervals between each sample, constant value in this case
        dts = np.diff(ts)
        # Muliply frequency samples by their time by 2PI, this is their offset, position
        #  on the curve of the periodic signal function
        dphis = PI2 * freqs[:-1] * dts
        # Take the cumulative sum at each sampling time, this is the total offset at each step
        # NOTE: equivalent to integration
        phases = np.cumsum(dphis)
        # Ensure the first sample is 0 to avoid discontinuity
        phases = np.insert(phases, 0, 0)
        # Convert the offset at each step to its angle in radians and multiply that by amp
        ys = np.cos(phases)
        ys = normalize(ys, self.amp)
        return ys


def run():
    trombone_glissando_sig = TromboneGlissandoSignal(START_FREQ, END_FREQ)
    wave = trombone_glissando_sig.make_wave(start=START, duration=DURATION_SECS,
                                            framerate=FRAMERATE)
    print('Plotting trombone glissando wave. Verify the waveform.')
    # wave.plot()
    print('Playing trombone glissando wave')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 3: ex_5_trombone_glissando.py")
    print("****************************")
    run()
