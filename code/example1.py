"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkdsp
import thinkplot

import matplotlib.pyplot as pyplot


def sample_violin(start=1.2, duration=0.6):
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    # extract a sample
    sample = wave.sample(start, duration)

    # plot the spectrum
    spectrum = sample.make_spectrum()

    spectrum.low_pass(600)

    filtered = spectrum.make_wave()
    filtered.normalize()
    filtered.apodize()

    sample.apodize()

    filename = 'filtered.wav'
    wfile = thinkdsp.WavFileWriter(filename, sample.framerate)
    wfile.write(sample)
    wfile.write(filtered)
    wfile.close()

    thinkdsp.play_wave(filename)


def main():
    sample_violin()
    return

    cos_sig = thinkdsp.CosSignal(freq=440, amp=1.0, offset=0)
    sin_sig = thinkdsp.SinSignal(freq=880, amp=0.5, offset=0)

    mix = sin_sig + cos_sig

    wave = mix.make_wave(duration=0.5, start=0, framerate=11025)
    print len(wave.ys)
    print 1.0 / 11025 * 1000

    period = mix.period
    sample = wave.sample(start=0, duration=period*3)

    sample.plot()
    thinkplot.Save(root='example1',
                   xlabel='time (s)',
                   axis=[0, period*3, -1.55, 1.55])

    wave.normalize()
    wave.apodize()
    wave.write(filename='example1.wav')

    thinkdsp.play_wave(filename='example1.wav', player='aplay')

    spectrum = wave.make_spectrum()
    spectrum.plot()
    thinkplot.show()


if __name__ == '__main__':
    main()
