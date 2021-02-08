# Create a CubicSignal, which is defined in thinkdsp. Compute the second
# difference by applying diff twice. What does the result look like?
# Compute the second derivative by applying differentiate to the spectrum twice.
# Does the result look the same? Plot the filters that correspond to the
# second difference and the second derivative and compare them.

import numpy as np

from code.thinkdsp import CubicSignal, Wave

FREQ_A4 = 440
AMP = 0.5
OFFSET = 0
FRAMERATE = 11025
START = 0
DURATION_SECS = 1


def run():
    # Create the cubic wave and plot it
    cubic_sig = CubicSignal(freq=FREQ_A4, amp=AMP, offset=OFFSET)
    wave = cubic_sig.make_wave(start=START, duration=DURATION_SECS, framerate=FRAMERATE)
    # wave.plot()

    # Compute the diff twice and plot it 
    diff = np.diff(wave.ys)
    diff_wave = Wave(diff, framerate=FRAMERATE)
    spectrum = diff_wave.make_spectrum()
    spectrum.plot()
    second_diff = np.diff(diff_wave.ys)
    second_diff_wave = Wave(second_diff, framerate=FRAMERATE)
    spectrum = second_diff_wave.make_spectrum()
    spectrum.plot()

    # Compute the spectrum of the cubic wave, differentiate, plot
    wave_spectrum = wave.make_spectrum()
    wave_spectrum = wave_spectrum.differentiate()
    wave_spectrum.plot()
    # Convert spectrum back to a wave and plot it.
    deriv_wave = wave_spectrum.make_wave()
    deriv_wave.plot()
    # Differentiate again and plot
    wave_spectrum = wave_spectrum.differentiate()
    wave_spectrum.plot()
    # Convert spectrum back to a wave and plot it.
    second_deriv = wave_spectrum.make_wave()
    second_deriv.plot()


if __name__ == '__main__':
    print("\nChapter 9: ex_2_triangle_diff_differentiate.py")
    print("****************************")
    run()
