# The algorithm in this chapter for generating pink noise is conceptually simple but
# computationally expensive. There are more efficient alternatives, like the Voss–McCartney
# algorithm. Research this method, implement it, compute the spectrum of the result, and
# confirm that it has the desired relationship between power and frequency.

import numpy as np
import pandas as pd
import code.thinkplot as thinkplot
from code.thinkdsp import Noise
from exercises.lib.lib import play_wave

FRAMERATE = 22050


class VossMcCartneyPinkNoise(Noise):
    """Represents pink noise generated using the Voss-McCartney algorithm."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        if self.amp > 1.0:
            raise ValueError('Amp must be 0 > amp <= 1.0')
        ys = VossMcCartneyPinkNoise._voss(len(ts))
        return ys

    @staticmethod
    def _voss(nrows, ncols=16):
        """
        Generates pink noise using the Voss-McCartney algorithm
        nrows - number of values to generate
        rcols - number of random sources to add
        returns -  NumPy array
        """
        array = np.full((nrows, ncols), np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)
        # the total number of changes is nrows
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)

        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)
        return total.values


def run():
    duration = 1.5
    amp = 0.5
    noise_signal = VossMcCartneyPinkNoise(amp=amp)
    wave = noise_signal.make_wave(start=0, duration=duration, framerate=FRAMERATE)
    play_wave(wave)
    print('Plotting Spectrum for VossMcCartneyPinkNoise')
    spectrum = wave.make_spectrum()
    # TODO Make my own make_spectrum wrapper to fix this
    # NOTE: Fix bug where first element of spectrum is very high and blows up chart scale
    spectrum.hs[0] = 0.
    spectrum.plot_power(high=np.max(spectrum.hs))
    thinkplot.config(xscale='log', yscale='log')


if __name__ == '__main__':
    print("\nChapter 4: ex_5_voss_mccartney_pink_noise.py")
    print("****************************")
    run()

