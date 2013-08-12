"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkdsp
import thinkplot

import matplotlib.pyplot as pyplot


def main():
    cos_sig = thinkdsp.CosSignal(freq=440, amp=1.0, offset=0)
    sin_sig = thinkdsp.SinSignal(freq=880, amp=0.5, offset=0)

    mix = sin_sig + cos_sig

    wave = mix.make_wave(duration=0.5, start=0, framerate=11025)

    period = mix.period
    sample = wave.sample(start=0, duration=period*3)

    sample.plot()
    #pyplot.show()

    wave.normalize()
    wave.apodize()
    thinkdsp.write_wave(wave, filename='sound.wav')

    thinkdsp.play_wave(filename='sound.wav', player='aplay')

if __name__ == '__main__':
    main()
