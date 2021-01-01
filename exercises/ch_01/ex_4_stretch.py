# Write a function called stretch that takes a Wave and a stretch factor and
# speeds up or slows down the wave by modifying ts and framerate.
# Hint: it should only take two lines of code.

from code.thinkdsp import SinSignal
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22050


def stretch(wave, stretch_factor=1.):
    wave.ts *= stretch_factor
    wave.framerate *= 1. / stretch_factor


def run():
    sin_sig = SinSignal(freq=FREQ_A4, amp=0.5, offset=0)
    start = 0.
    duration_secs = 1.
    wave = sin_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing wave before stretching')
    play_wave(wave)

    print('Playing wave after stretching to slow down')
    stretch(wave, 2)
    play_wave(wave)

    print('Playing wave after stretching to speed up')
    stretch(wave, 0.4)
    play_wave(wave)


if __name__ == '__main__':
    run()
