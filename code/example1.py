"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkdsp
import thinkplot


def mix_cosines():
    """Demonstrates methods in the thinkdsp module.
    """

    # create a SumSignal
    cos_sig = thinkdsp.CosSignal(freq=440, amp=1.0, offset=0)
    sin_sig = thinkdsp.SinSignal(freq=880, amp=0.5, offset=0)

    mix = sin_sig + cos_sig

    # create a wave
    wave = mix.make_wave(duration=0.5, start=0, framerate=11025)
    print 'Number of samples', len(wave.ys)
    print 'Timestep in ms', 1.0 / 11025 * 1000

    # select a segment
    period = mix.period
    segment = wave.segment(start=0, duration=period*3)

    # plot the segment
    segment.plot()
    thinkplot.Save(root='example1',
                   xlabel='time (s)',
                   axis=[0, period*3, -1.55, 1.55])

    # write the whole wave
    wave.normalize()
    wave.apodize()
    wave.write(filename='example1.wav')

    # play the wave
    thinkdsp.play_wave(filename='example1.wav', player='aplay')


def violin_example(start=1.2, duration=0.6):
    """Demonstrates methods in the thinkdsp module.
    """
    # read the violin recording
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    # extract a segment
    segment = wave.segment(start, duration)

    # make the spectrum
    spectrum = segment.make_spectrum()

    # apply a filter
    spectrum.low_pass(600)

    # invert the spectrum
    filtered = spectrum.make_wave()

    # prepare the original and filtered segments
    filtered.normalize()
    filtered.apodize()
    segment.apodize()

    # write the original and filtered segments to a file
    filename = 'filtered.wav'
    wfile = thinkdsp.WavFileWriter(filename, segment.framerate)
    wfile.write(segment)
    wfile.write(filtered)
    wfile.close()

    thinkdsp.play_wave(filename)


def main():
    mix_cosines()
    violin_example()


if __name__ == '__main__':
    main()
