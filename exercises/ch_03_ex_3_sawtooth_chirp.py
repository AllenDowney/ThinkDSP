# Write a class called SawtoothChirp that extends Chirp and overrides evaluate to
# generate a sawtooth waveform with frequency that increases (or decreases) linearly.
# Draw a sketch of what you think the spectrogram of this signal looks like, and then plot it. 
# The effect of aliasing should be visually apparent, and if you listen, you can hear it.

from code.thinkdsp import Sinusoid
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22500


class SawtoothChirpSignal(Sinusoid):
    def evaluate(self, ts):
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        frac -= 0.5
        frac = unbias(frac)
        # Map negative offsets to -1, like the negative half of a square wave
        # Just unbias positive values, making the positive half like the triangle wave
        # Use the resulting mapping of all offsets to scale amp for the signal instance
        ys = np.where(frac >= 0.0, normalize(frac, self.amp), -1.0)
        return ys


def run():
    pass


if __name__ == '__main__':
    print("\nChapter 3: ex_2_sawtooth_chirp.py")
    print("****************************")
    run()

