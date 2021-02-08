# explore the effects of cumsum and integrate on a signal. Create a square
# wave and plot it. Apply cumsum and plot the result. Compute the
# spectrum of the square wave, apply integrate, and plot the result.
# Convert the spectrum back to a wave and plot it. Are there differences
# between the effects of cumsum and integrate for this wave?

import numpy as np

from code.thinkdsp import SquareSignal, Wave

FREQ_A4 = 440
AMP = 0.5
OFFSET = 0
FRAMERATE = 11025
START = 0
DURATION_SECS = 1


def run():
    # Create the triangle wave and plot it
    square_sig = SquareSignal(freq=FREQ_A4, amp=AMP, offset=OFFSET)
    wave = square_sig.make_wave(start=START, duration=DURATION_SECS, framerate=FRAMERATE)
    wave.plot()

    # Compute the cumsum of the wave, compute the spectrum of the cumsum, plot it
    cumsum_wave = wave.cumsum()
    cumsum_wave.unbias()
    cumsum_wave.plot()
    spectrum = cumsum_wave.make_spectrum()
    spectrum.plot()

    # Compute the spectrum of the wave, integrate, plot
    wave_spectrum = wave.make_spectrum()
    wave_spectrum.integrate()
    wave_spectrum.plot()

    # Convert spectrum back to a wave and plot it
    integ = wave_spectrum.make_wave()
    integ.plot()


if __name__ == '__main__':
    print("\nChapter 9: ex_2_triangle_diff_differentiate.py")
    print("****************************")
    run()

