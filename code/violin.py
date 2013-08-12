"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkdsp
import thinkplot


def sample_violin(start=1.2, duration=0.6):
    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    # extract a sample
    sample = wave.sample(start, duration)
    sample.normalize()
    sample.apodize()
    thinkdsp.write_wave(sample, 'violin_sample1.wav')

    # plot the spectrum
    spectrum = sample.spectrum()
    n = len(spectrum.hs)
    spectrum.plot(high=n/2)
    thinkplot.Save(root='violin2',
                   xlabel='frequency (Hz)',
                   ylabel='amplitude density')

    # print the top 5 peaks
    peaks = spectrum.peaks()
    for amp, freq in peaks[:10]:
        print freq, amp

    # compare the samples to a 440 Hz Triangle wave
    note = thinkdsp.make_note(69, 0.6, 
                              sig_cons=thinkdsp.TriangleSignal, 
                              framerate=sample.framerate)

    wfile = thinkdsp.WavFileWriter('violin_sample2.wav', note.framerate)
    wfile.write(note)
    wfile.write(sample)
    wfile.write(note)
    wfile.close()


def sin_spectrum():
    wave = thinkdsp.make_note(69, 0.5, SinSignal)
    spectrum = wave.spectrum()
    spectrum.plot()
    thinkplot.Show()

    peaks = spectrum.peaks()
    print peaks[0]

    wave2 = spectrum.wave()

    wave2.plot()
    thinkplot.Show()

    thinkdsp.write_wave(wave2)


def plot_sinusoid(duration = 0.00685):
    """Plots three cycles of a 440 Hz sinusoid.

    duration: float
    """
    signal = thinkdsp.SinSignal(440)
    wave = signal.make_wave(duration, framerate=44100)
    wave.plot()
    thinkplot.Save(root='sinusoid1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def plot_violin(start=1.30245, duration=0.00683):
    """Plots three cycles of a sampled violin playing A4.

    duration: float
    """
    period = duration/3
    freq = 1/period
    print freq

    wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')

    sample = wave.sample(start, duration)
    sample.normalize()

    sample.plot()
    thinkplot.Save(root='violin1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def plot_tuning(start=7.0, duration=0.006835):
    """Plots three cycles of a sampled violin playing A4.

    duration: float
    """
    period = duration/3
    freq = 1/period
    print period, freq

    wave = thinkdsp.read_wave('18871__zippi1__sound-bell-440hz.wav')

    sample = wave.sample(start, duration)
    sample.normalize()

    sample.plot()
    thinkplot.Save(root='tuning1',
                   xlabel='time (s)',
                   axis=[0, duration, -1.05, 1.05])


def main():
    sample_violin()
    return

    plot_tuning()
    return

    plot_sinusoid()
    plot_violin()



if __name__ == '__main__':
    main()
