# Write a class called SawtoothChirp that extends Chirp and overrides evaluate to
# generate a sawtooth waveform with frequency that increases (or decreases) linearly.
# Draw a sketch of what you think the spectrogram of this signal looks like, and then plot it. 
# The effect of aliasing should be visually apparent, and if you listen, you can hear it.

import numpy as np

from code.thinkdsp import PI2, Sinusoid, normalize, unbias
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22500
CHIRP_START_FREQ = 220
CHIRP_END_FREQ = 880
OFFSET = 0
START = 0
DURATION_SECS = 1
AMP = 0.5


class SawtoothSignal(Sinusoid):
    def __init__(self, **options):
        super(SawtoothSignal, self).__init__(**options)
        self.ts = None
        self.ys = None

    # For more comments on this logic see ch_02/ex_2_sawtooth.py
    def evaluate(self, ts):
        # Capture ts and ys for use by SawtoothChirpSignal#evaluate()
        self.ts = ts
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        frac -= 0.5
        ys = np.where(frac > 0.0, unbias(frac), -0.5)
        self.ys = ys
        # Don't normalize by self.amp because this is used in combination
        # with chirp and we don't want to normalize twice
        return ys


# Chirp implementation From thinkdsp
class SawtoothChirpSignal(Sinusoid):
    def __init__(self, chirp_start_freq, chirp_end_freq, sawtooth_freq,
                 framerate=None, **options):
        super(SawtoothChirpSignal, self).__init__(**options)
        self._chirp_start_freq = chirp_start_freq
        self._chirp_end_freq = chirp_end_freq
        self._sawtooth_sig = SawtoothSignal(freq=sawtooth_freq, amp=AMP, offset=OFFSET)
        self._sawtooth_sig.make_wave(start=START, duration=DURATION_SECS,
                                     framerate=framerate or FRAMERATE)

    def evaluate(self, ts):
        # Sample frequencies at even intervals from start to end
        freqs = np.linspace(self._chirp_start_freq, self._chirp_end_freq, len(ts) - 1)
        # Get the intervals between each sample, constant value in this case
        dts = np.diff(ts)
        # Muliply frequency samples by their time by 2PI, this is their offset, position
        #  on the curve of the periodic signal function
        dphis = PI2 * freqs * dts
        # Take the cumulative sum at each sampling time, this is the total offset at each step
        # NOTE: equivalent to integration
        # Notice also that this step is supralinear because its cumulative sum so it causes
        #  the pitch to rise, go up more quickly than moving linearly over each time step
        phases = np.cumsum(dphis)
        # Ensure the first sample is 0 to avoid discontinuity
        phases = np.insert(phases, 0, 0)
        # Convert the offset at each step to its angle in radians and multiply that by amp
        ys = np.cos(phases)
        # Multiply the samples for the chirp by the samples for the sawtooth. This effectively
        #  shapes the sawtooth by the shape of the chirp.
        ys *= self._sawtooth_sig.ys
        ys = normalize(ys, self.amp)
        return ys


def run():
    sawtooth_chirp_sig = SawtoothChirpSignal(CHIRP_START_FREQ, CHIRP_END_FREQ, FREQ_A4)
    # Note this actually just plots three periods of the wave
    wave = sawtooth_chirp_sig.make_wave(start=START, duration=DURATION_SECS, framerate=FRAMERATE)
    print('Plotting sawtooth chirp wave. Verify the waveform.')
    wave.plot()
    print('Playing sawtooth chirp wave')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 3: ex_2_sawtooth_chirp.py")
    print("****************************")
    run()

