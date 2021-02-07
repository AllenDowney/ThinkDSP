# Create a triangle wave and plot it. Apply diff and plot the result. Compute the spectrum of the
# triangle wave, apply differentiate, and plot the result. Convert the spectrum back to a wave and
# plot it. Are there differences between the effects of diff and differentiate for this wave?

import numpy as np

from code.thinkdsp import TriangleSignal, Wave

FREQ_A4 = 440
AMP = 0.5
OFFSET = 0
FRAMERATE = 11025
START = 0
DURATION_SECS = 1


def run():
    # Create the triangle wave and plot it
    triangle_sig = TriangleSignal(freq=FREQ_A4, amp=AMP, offset=OFFSET)
    wave = triangle_sig.make_wave(start=START, duration=DURATION_SECS, framerate=FRAMERATE)
    wave.plot()

    # Compute the diff of the triangle wave, compute the spectrum of the diff, plot it
    diff = np.diff(wave.ys)
    diff_wave = Wave(diff, framerate=FRAMERATE)
    spectrum = diff_wave.make_spectrum()
    spectrum.plot()

    # Compute the spectrum of the triangle wave, differentiate, plot
    wave_spectrum = wave.make_spectrum()
    wave_spectrum.differentiate()
    wave_spectrum.plot()

    # Convert spectrum back to a wave and plot it
    deriv = wave_spectrum.make_wave()
    deriv.plot()


if __name__ == '__main__':
    print("\nChapter 9: ex_2_triangle_diff_differentiate.py")
    print("****************************")
    run()
