# Create a sawtooth wave, compute its spectrum, then apply integrate twice. Plot the resulting
# wave and its spectrum. What is the mathematical form of the wave? Why does it resemble a sinusoid?

import numpy as np

from code.thinkdsp import SawtoothSignal, Wave

FREQ_A4 = 440
AMP = 0.5
OFFSET = 0
FRAMERATE = 11025
START = 0
DURATION_SECS = 1


def run():
    # Create a sawtooth wave
    sawtooth_sig = SawtoothSignal(freq=FREQ_A4, amp=AMP, offset=OFFSET)
    wave = sawtooth_sig.make_wave()
    wave.plot()

    # Compute its spectrum
    spectrum = wave.make_spectrum()
    spectrum.plot()

    # Integrte once and plot
    spectrum_integ_once = spectrum.copy().integrate()
    spectrum_integ_once.hs[0] = 0
    spectrum_integ_once.plot()
    spectrum_integ_once_wave = spectrum_integ_once.make_wave()
    spectrum_integ_once_wave.plot()

    # Integrate again and plot
    spectrum_integ_twice = spectrum_integ_once.integrate()
    spectrum_integ_twice.hs[0] = 0
    spectrum_integ_twice.plot()
    spectrum_integ_twice_wave = spectrum_integ_twice.make_wave()
    spectrum_integ_twice_wave.plot()


if __name__ == '__main__':
    print("\nChapter 9: ex_2_triangle_diff_differentiate.py")
    print("****************************")
    run()
