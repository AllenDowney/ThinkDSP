"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2015 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot
import math
import numpy as np

import matplotlib.pyplot as plt

PI2 = 2 * math.pi

FORMATS = ['pdf', 'eps']
FORMATS = ['pdf']


def plot_beeps():
    wave = thinkdsp.read_wave('253887__themusicalnomad__positive-beeps.wav')
    wave.normalize()

    thinkplot.preplot(3)

    # top left
    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2)
    plt.setp(ax1.get_xticklabels(), visible=False)

    wave.plot()
    thinkplot.config(title='Input waves', legend=False)

    # bottom left
    imp_sig = thinkdsp.Impulses([0.01, 0.4, 0.8, 1.2], 
                                amps=[1, 0.5, 0.25, 0.1])
    impulses = imp_sig.make_wave(start=0, duration=1.3, 
                                 framerate=wave.framerate)

    ax2 = plt.subplot2grid((4, 2), (2, 0), rowspan=2, sharex=ax1)
    impulses.plot()
    thinkplot.config(xlabel='Time (s)')

    # center right
    convolved = wave.convolve(impulses)

    ax3 = plt.subplot2grid((4, 2), (1, 1), rowspan=2)
    plt.title('Convolution')
    convolved.plot()
    thinkplot.config(xlabel='Time (s)')

    thinkplot.save(root='sampling1',
                   formats=FORMATS,
                   legend=False)


def plot_am():
    wave = thinkdsp.read_wave('105977__wcfl10__favorite-station.wav')
    wave.unbias()
    wave.normalize()

    # top
    ax1 = thinkplot.preplot(6, rows=4)
    spectrum = wave.make_spectrum(full=True)
    spectrum.plot(label='wave')

    xlim = [-22050, 22050]
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    #second
    carrier_sig = thinkdsp.CosSignal(freq=10000)
    carrier_wave = carrier_sig.make_wave(duration=wave.duration, 
                                         framerate=wave.framerate)
    modulated = wave * carrier_wave

    ax2 = thinkplot.subplot(2, sharey=ax1)
    modulated.make_spectrum(full=True).plot(label='modulated')
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    # third
    demodulated = modulated * carrier_wave
    demodulated_spectrum = demodulated.make_spectrum(full=True)

    ax3 = thinkplot.subplot(3, sharey=ax1)
    demodulated_spectrum.plot(label='demodulated')
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    #fourth
    ax4 = thinkplot.subplot(4, sharey=ax1)
    demodulated_spectrum.low_pass(10000)
    demodulated_spectrum.plot(label='filtered')
    thinkplot.config(xlim=xlim, xlabel='Frequency (Hz)')

    thinkplot.save(root='sampling2',
                   formats=FORMATS)

    #carrier_spectrum = carrier_wave.make_spectrum(full=True)
    #carrier_spectrum.plot()


    #convolved = spectrum.convolve(carrier_spectrum)
    #convolved.plot()


    #reconvolved = convolved.convolve(carrier_spectrum)
    #reconvolved.plot()


def sample(wave, factor):
    """Simulates sampling of a wave.
    
    wave: Wave object
    factor: ratio of the new framerate to the original
    """
    ys = np.zeros(len(wave))
    ys[::factor] = wave.ys[::factor]
    ts = wave.ts[:]
    return thinkdsp.Wave(ys, ts, wave.framerate) 


def make_impulses(wave, factor):
    ys = np.zeros(len(wave))
    ys[::factor] = 1
    ts = np.arange(len(wave)) / wave.framerate
    return thinkdsp.Wave(ys, ts, wave.framerate)


def plot_segments(original, filtered):
    start = 1
    duration = 0.01
    original.segment(start=start, duration=duration).plot(color='gray')
    filtered.segment(start=start, duration=duration).plot()


def plot_amen():
    wave = thinkdsp.read_wave('263868__kevcio__amen-break-a-160-bpm.wav')
    wave.normalize()

    ax1 = thinkplot.preplot(2, rows=2)
    wave.make_spectrum(full=True).plot()

    xlim = [-22050, 22050]
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    ax2 = thinkplot.subplot(2, sharey=ax1)
    sampled = sample(wave, 4)
    sampled.make_spectrum(full=True).plot()
    thinkplot.config(xlim=xlim, xlabel='Frequency (Hz)')

    thinkplot.show()
    return


def plot_amen2():
    wave = thinkdsp.read_wave('263868__kevcio__amen-break-a-160-bpm.wav')
    wave.normalize()

    ax1 = thinkplot.preplot(6, rows=4)
    wave.make_spectrum(full=True).plot(label='spectrum')
    xlim = [-22050-250, 22050]
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    ax2 = thinkplot.subplot(2)
    impulses = make_impulses(wave, 4)
    impulses.make_spectrum(full=True).plot(label='impulses')
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    ax3 = thinkplot.subplot(3)
    sampled = wave * impulses
    spectrum = sampled.make_spectrum(full=True)
    spectrum.plot(label='sampled')
    thinkplot.config(xlim=xlim, xticklabels='invisible')

    ax4 = thinkplot.subplot(4)
    spectrum.low_pass(5512.5)
    spectrum.plot(label='filtered')
    thinkplot.config(xlim=xlim, xlabel='Frequency (Hz)')

    thinkplot.save(root='sampling4',
                   formats=FORMATS)


def plot_amen3():
    filtered = spectrum.make_wave()

    plot_segments(wave, filtered)


def make_boxcar(spectrum, factor):
    """Makes a boxcar filter for the given spectrum.
    
    spectrum: Spectrum to be filtered
    factor: sampling factor
    """
    fs = np.copy(spectrum.fs)
    hs = np.zeros_like(spectrum.hs)
    
    cutoff = spectrum.framerate / 2 / factor
    for i, f in enumerate(fs):
        if abs(f) <= cutoff:
            hs[i] = 1
    return thinkdsp.Spectrum(hs, fs, spectrum.framerate, full=spectrum.full)


def plot_bass():
    wave = thinkdsp.read_wave('328878__tzurkan__guitar-phrase-tzu.wav')
    wave.normalize()
    wave.plot()

    wave.make_spectrum(full=True).plot()


    sampled = sample(wave, 4)
    sampled.make_spectrum(full=True).plot()

    spectrum = sampled.make_spectrum(full=True)
    boxcar = make_boxcar(spectrum, 4)
    boxcar.plot()

    filtered = (spectrum * boxcar).make_wave()
    filtered.scale(4)
    filtered.make_spectrum(full=True).plot()

    plot_segments(wave, filtered)


    diff = wave.ys - filtered.ys
    #thinkplot.plot(diff)
    np.mean(abs(diff))

    sinc = boxcar.make_wave()
    ys = np.roll(sinc.ys, 50)
    thinkplot.plot(ys[:100])


def plot_sinc_demo(wave, factor, start=None, duration=None):

    def make_sinc(t, i, y):
        """Makes a shifted, scaled copy of the sinc function."""
        sinc = boxcar.make_wave()
        sinc.shift(t)
        sinc.roll(i)
        sinc.scale(y * factor)
        return sinc
 
    def plot_sincs(wave):
        """Plots sinc functions for each sample in wave."""
        t0 = wave.ts[0]
        for i in range(0, len(wave), factor):
            sinc = make_sinc(t0, i, wave.ys[i])
            seg = sinc.segment(start, duration)
            seg.plot(color='green', linewidth=0.5, alpha=0.3)
            if i == 0:
                total = sinc
            else:
                total += sinc
            
        seg = total.segment(start, duration)        
        seg.plot(color='blue', alpha=0.5)

    sampled = sample(wave, factor)
    spectrum = sampled.make_spectrum()
    boxcar = make_boxcar(spectrum, factor)

    start = wave.start if start is None else start
    duration = wave.duration if duration is None else duration
        
    sampled.segment(start, duration).plot_vlines(color='gray')
    wave.segment(start, duration).plot(color='gray')
    plot_sincs(wave)


def plot_sincs():
    start = 1.0
    duration = 0.01
    factor = 4

    short = wave.segment(start=start, duration=duration)
    short.plot()

    sampled = sample(short, factor)
    sampled.plot_vlines(color='gray')


    spectrum = sampled.make_spectrum()
    boxcar = make_boxcar(spectrum, factor)
    sinc = boxcar.make_wave()
    sinc.shift(sampled.ts[0])
    sinc.roll(len(sinc)//2)

    thinkplot.preplot(1)
    sinc.plot()

    # CAUTION: don't call plot_sinc_demo with a large wave or it will
    # fill memory and crash
    plot_sinc_demo(short, 4)

    start = short.start + 0.004
    duration = 0.00061
    plot_sinc_demo(short, 4, start, duration)
    thinkplot.config(xlim=[start, start+duration],
                         ylim=[-0.05, 0.17], legend=False)


def main():
    #plot_beeps()
    #plot_am()
    #plot_amen()
    plot_amen2()


if __name__ == '__main__':
    main()
