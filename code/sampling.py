"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2015 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkdsp
import thinkplot

import numpy as np
import matplotlib.pyplot as plt

PI2 = 2 * np.pi
FORMATS = ['pdf', 'eps']


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

XLIM = [-22050, 22050]

def plot_am():
    wave = thinkdsp.read_wave('105977__wcfl10__favorite-station.wav')
    wave.unbias()
    wave.normalize()

    # top
    ax1 = thinkplot.preplot(6, rows=4)
    spectrum = wave.make_spectrum(full=True)
    spectrum.plot(label='spectrum')

    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    #second
    carrier_sig = thinkdsp.CosSignal(freq=10000)
    carrier_wave = carrier_sig.make_wave(duration=wave.duration, 
                                         framerate=wave.framerate)
    modulated = wave * carrier_wave

    ax2 = thinkplot.subplot(2, sharey=ax1)
    modulated.make_spectrum(full=True).plot(label='modulated')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    # third
    demodulated = modulated * carrier_wave
    demodulated_spectrum = demodulated.make_spectrum(full=True)

    ax3 = thinkplot.subplot(3, sharey=ax1)
    demodulated_spectrum.plot(label='demodulated')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    #fourth
    ax4 = thinkplot.subplot(4, sharey=ax1)
    demodulated_spectrum.low_pass(10000)
    demodulated_spectrum.plot(label='filtered')
    thinkplot.config(xlim=XLIM, xlabel='Frequency (Hz)')

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


def plot_sampling(wave, root):
    ax1 = thinkplot.preplot(2, rows=2)
    wave.make_spectrum(full=True).plot(label='spectrum')

    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    ax2 = thinkplot.subplot(2)
    sampled = sample(wave, 4)
    sampled.make_spectrum(full=True).plot(label='sampled')
    thinkplot.config(xlim=XLIM, xlabel='Frequency (Hz)')

    thinkplot.save(root=root,
                   formats=FORMATS)


def plot_sampling2(wave, root):
    ax1 = thinkplot.preplot(6, rows=4)
    wave.make_spectrum(full=True).plot(label='spectrum')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    ax2 = thinkplot.subplot(2)
    impulses = make_impulses(wave, 4)
    impulses.make_spectrum(full=True).plot(label='impulses')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    ax3 = thinkplot.subplot(3)
    sampled = wave * impulses
    spectrum = sampled.make_spectrum(full=True)
    spectrum.plot(label='sampled')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    ax4 = thinkplot.subplot(4)
    spectrum.low_pass(5512.5)
    spectrum.plot(label='filtered')
    thinkplot.config(xlim=XLIM, xlabel='Frequency (Hz)')

    thinkplot.save(root=root,
                   formats=FORMATS)


def plot_sampling3(wave, root):
    ax1 = thinkplot.preplot(6, rows=3)
    wave.make_spectrum(full=True).plot(label='spectrum')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    impulses = make_impulses(wave, 4)

    ax2 = thinkplot.subplot(2)
    sampled = wave * impulses
    spectrum = sampled.make_spectrum(full=True)
    spectrum.plot(label='sampled')
    thinkplot.config(xlim=XLIM, xticklabels='invisible')

    ax3 = thinkplot.subplot(3)
    spectrum.low_pass(5512.5)
    spectrum.plot(label='filtered')
    thinkplot.config(xlim=XLIM, xlabel='Frequency (Hz)')

    thinkplot.save(root=root,
                   formats=FORMATS)

    #filtered = spectrum.make_wave()
    #plot_segments(wave, filtered)


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



def plot_sinc_demo(wave, factor, start=None, duration=None):

    def make_sinc(t, i, y):
        """Makes a shifted, scaled copy of the sinc function."""
        sinc = boxcar.make_wave()
        sinc.shift(t)
        sinc.roll(i)
        sinc.scale(y * factor)
        return sinc
 
    def plot_mini_sincs(wave):
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
    plot_mini_sincs(wave)


def plot_sincs(wave):
    start = 1.0
    duration = 0.01
    factor = 4

    short = wave.segment(start=start, duration=duration)
    #short.plot()

    sampled = sample(short, factor)
    #sampled.plot_vlines(color='gray')

    spectrum = sampled.make_spectrum(full=True)
    boxcar = make_boxcar(spectrum, factor)

    sinc = boxcar.make_wave()
    sinc.shift(sampled.ts[0])
    sinc.roll(len(sinc)//2)

    thinkplot.preplot(2, cols=2)
    sinc.plot()
    thinkplot.config(xlabel='Time (s)')

    thinkplot.subplot(2)
    boxcar.plot()
    thinkplot.config(xlabel='Frequency (Hz)',
                     ylim=[0, 1.05],
                     xlim=[-boxcar.max_freq, boxcar.max_freq])

    thinkplot.save(root='sampling6',
                   formats=FORMATS)

    return

    # CAUTION: don't call plot_sinc_demo with a large wave or it will
    # fill memory and crash
    plot_sinc_demo(short, 4)
    thinkplot.config(xlabel='Time (s)')
    thinkplot.save(root='sampling7',
                   formats=FORMATS)

    start = short.start + 0.004
    duration = 0.00061
    plot_sinc_demo(short, 4, start, duration)
    thinkplot.config(xlabel='Time (s)',
                     xlim=[start, start+duration],
                     ylim=[-0.06, 0.17], legend=False)
    thinkplot.save(root='sampling8',
                   formats=FORMATS)


def kill_yticklabels():
    axis = plt.gca()
    plt.setp(axis.get_yticklabels(), visible=False)


def show_impulses(wave, factor, i):
    thinkplot.subplot(i)
    thinkplot.preplot(2)
    impulses = make_impulses(wave, factor)
    impulses.segment(0, 0.001).plot_vlines(linewidth=2, xfactor=1000)
    if i == 1:
        thinkplot.config(title='Impulse train',
                         ylim=[0, 1.05])
    else:
        thinkplot.config(xlabel='Time (ms)',
                         ylim=[0, 1.05])
    
    thinkplot.subplot(i+1)
    impulses.make_spectrum(full=True).plot()
    kill_yticklabels()
    if i == 1:
        thinkplot.config(title='DFT of impulse train', 
                         xlim=[-22400, 22400])
    else:
        thinkplot.config(xlabel='Frequency (Hz)',
                         xlim=[-22400, 22400])


def plot_impulses(wave):
    thinkplot.preplot(rows=2, cols=2)
    show_impulses(wave, 4, 1)
    show_impulses(wave, 8, 3)
    thinkplot.save('sampling9',
                   formats=FORMATS)


def main():
    wave = thinkdsp.read_wave('328878__tzurkan__guitar-phrase-tzu.wav')
    wave.normalize()

    plot_sampling3(wave, 'sampling5')
    plot_sincs(wave)

    plot_beeps()
    plot_am()

    wave = thinkdsp.read_wave('263868__kevcio__amen-break-a-160-bpm.wav')
    wave.normalize()
    plot_impulses(wave)

    plot_sampling(wave, 'sampling3')
    plot_sampling2(wave, 'sampling4')


if __name__ == '__main__':
    main()
