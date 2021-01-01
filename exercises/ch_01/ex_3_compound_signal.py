# Exercise 1-3.  Synthesize a compound signal by creating SinSignal and
# CosSignal objects and adding them up.
# Evaluate the signal to get a Wave, and listen to it.
# Compute its Spectrum and plot it.
# What happens if you add frequency components that are not multiples of the fundamental?

from code.thinkdsp import read_wave, SinSignal, CosSignal
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FREQ_A5 = 880
FRAMERATE = 22050


def run():
    cos_sig = CosSignal(freq=FREQ_A4, amp=1.0, offset=0)
    sin_sig = SinSignal(freq=FREQ_A5, amp=0.5, offset=0)
    compound_sig = sin_sig + cos_sig

    start = 0.
    duration_secs = 1.
    wave = compound_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing compound sin + cos signal with signal frequencies at A4 and A5')
    play_wave(wave)

    # Now add a signal that isn't a multiple of the fundamental signal frequency
    dissonant_sig = SinSignal(freq=FREQ_A5 - 25, amp=0.5, offset=0)
    compound_sig += dissonant_sig
    wave = compound_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing compound sin + cos signal + dissonant_sig')
    play_wave(wave)


if __name__ == '__main__':
    run()
