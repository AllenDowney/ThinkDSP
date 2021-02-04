# In addition to the Gaussian window we used in this chapter, create a Hamming window with the same
# size. Zero-pad the windows and plot their DFTs. Which window acts as a better low-pass filter?
# You might find it useful to plot the DFTs on a log-y scale.

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

from code.thinkdsp import read_wave, zero_pad
import code.thinkplot as thinkplot

SOUND_FILE = 'exercises/ch_01/SIG_126_A_Retro_Synth.wav'
FRAMERATE = 22050


def smooth(ys, window):
    print('Smoothing ...')
    N = len(ys)
    smoothed = np.zeros(N)
    padded = zero_pad(window, N)
    rolled = padded

    for i in range(N):
        smoothed[i] = sum(rolled * ys)
        rolled = np.roll(rolled, 1)
    return smoothed


def _plot(signal, title):
    plt.title(title)
    plt.ylabel('Signal Frequency')
    plt.xlabel('Time')
    plt.yscale('log')
    plt.plot(signal, color='blue')
    plt.show()


def _make_normalized_gaussian_window(window_size, std):
    gaussian = signal.windows.gaussian(M=window_size, std=std, sym=True)
    gaussian /= sum(gaussian)
    return gaussian


def run():
    wave = read_wave(SOUND_FILE)
    start = 0
    duration = 0.1
    segment = wave.segment(start, duration)

    stds = range(1, 10)
    window_sizes = [10, 25, 100]

    # M - number of points in the window
    # std - standard deviation
    # sym - flag for a symmetris window or not. Symmetrics is better for filters. 
    # TODO - why does he not use sym = True in the book?
    for window_size in window_sizes:
        for std in stds:
            gaussian = _make_normalized_gaussian_window(window_size, std)
            fft_gaussian = np.fft.fft(gaussian)
            _plot(fft_gaussian, f'Gaussian Window Signal std_dev = {std} size = {window_size}')
            _plot(segment.ys, 'Wave')
            smoothed_wave = smooth(segment.ys, gaussian)
            _plot(smoothed_wave, f'Wave Smoothed by Gaussian Window std_dev = {std} size = {window_size}')

    for window_size in window_sizes:
        hamming = np.hamming(window_size)
        _plot(fft_gaussian, f'Hamming Window size = {window_size}')
        _plot(segment.ys, 'Wave')
        smoothed_wave = smooth(segment.ys, hamming)
        _plot(smoothed_wave, f'Wave Smoothed by Hamming Window size = {window_size}')


if __name__ == '__main__':
    print("\nChapter 8: ex_3_compare_gaussian_hamming_window.py")
    print("****************************")
    run()
