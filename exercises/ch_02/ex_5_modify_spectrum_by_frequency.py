# Write a function that takes a Spectrum as a parameter and modifies it by dividing
#  each element of hs by the corresponding frequency from fs.
#  Hint: since division by zero is undefined, you might want to set spectrum.hs[0] = 0.

# Test your function using a square, triangle, or sawtooth wave:
#  - Compute the Spectrum and plot it.
#  - Modify the Spectrum using your function and plot it again.
#  - Use Spectrum.make_wave to make a Wave from the modified Spectrum, and listen to it.
#  What effect does this operation have on the signal? 

import numpy as np

from code.thinkdsp import TriangleSignal, decorate
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22500


def modify_spectrum_by_frequency(spectrum):
    # hs is an array of numpy complex numbers, where the real part represents amplitude
    #  and the imaginary part represents phase offset in radians.
    # fs is a corresponding numpy array of signal frequencies
    spectrum.hs /= np.where(spectrum.fs > 0.0, spectrum.fs, 1.)


def run():
    triangle_sig = TriangleSignal(freq=FREQ_A4, amp=0.5, offset=0)

    start = 0.
    duration_secs = 1.
    segment = triangle_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)

    spectrum = segment.make_spectrum()
    wave = spectrum.make_wave()
    print('Plotting unmodified triangle signal spectrum')
    wave.plot()
    decorate(xlabel='Time (s)')
    print('Playing unmodified triangle signal spectrum')
    play_wave(wave)

    modify_spectrum_by_frequency(spectrum)
    wave = spectrum.make_wave()
    print('Plotting triangle signal spectrum modified by frequency')
    wave.plot()
    decorate(xlabel='Time (s)')
    print('Playing triangle signal spectrum modified by frequency')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 2: ex_5_modify_spectrum_by_frequency.py")
    print("****************************")
    run()
