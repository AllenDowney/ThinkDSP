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
    print len(wave.ys)
    print 1.0 / 11025 * 1000

    period = mix.period
    sample = wave.sample(start=0, duration=period*3)

    sample.plot()
    #pyplot.show()
    thinkplot.Save(root='example1',
                   xlabel='time (s)',
                   axis=[0, period*3, -1.55, 1.55])

    wave.normalize()
    wave.apodize()
    wave.write(filename='example1.wav')

    thinkdsp.play_wave(filename='example1.wav', player='aplay')


if __name__ == '__main__':
    main()
