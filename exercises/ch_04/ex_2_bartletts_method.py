 # In a noise signal, the mixture of frequencies changes over time. In the long run, we
 #  expect the power at all frequencies to be equal, but in any sample, the power at
 #  each frequency is random.
 # To estimate the long-term average power at each frequency, we can break a long signal
 #  into segments, compute the power spectrum for each segment, and then compute the
 #  average across the segments.
 # You can read more about this algorithm at http://en.wikipedia.org/wiki/Bartlett’s_method.
 # Implement Bartlett’s method and use it to estimate the power spectrum for a noise wave.
 # Hint: look at the implementation of make_spectrogram.

# From Wikipedia
# The original N point data segment is split up into K (non-overlapping) data segments,
#  each of length M
# For each segment, compute the periodogram by computing the discrete Fourier transform
#  then computing the squared magnitude of the result and dividing this by M.
# Average the result of the periodograms above for the K data segments.

import matplotlib.pyplot as plt
from code.thinkdsp import UncorrelatedGaussianNoise, read_wave


SOUND_FILE = 'exercises/ch_01/SIG_126_A_Retro_Synth.wav'
FRAMERATE = 22050


def bartletts(wave, num_segments):
    step = len(wave.ys) // num_segments
    segment_vals = []
    start = 0
    end = start + step
    while end < len(wave.ys):
        segment = wave.slice(start, end)
        # Spectrum is a DFT of the wave
        spectrum = segment.make_spectrum()
        segment_vals.append((spectrum.real ** 2) / step)
        start += step
        end += step
    return sum(segment_vals) / num_segments


def _plot(periodogram, title):
    plt.title(title)
    plt.ylabel('Signal Frequency')
    plt.xlabel('Time')
    plt.plot(periodogram, color='blue')
    plt.show()


def run():
    start = 0.
    duration = 1.5  # secs
    num_segments = 150

    wave = read_wave(SOUND_FILE)
    # Smooth the boundaries of the wave to reduce frequency artifacts that would distort
    # the spectrum
    wave.hamming()
    segment = wave.segment(start, duration)
    periodogram = bartletts(segment, num_segments)
    _plot(periodogram, 'Bartlett\'s Method Periodogram - Non-Noise Wave')

    noise_signal = UncorrelatedGaussianNoise()
    wave = noise_signal.make_wave(duration, FRAMERATE)
    wave.hamming()
    periodogram = bartletts(wave, num_segments)
    _plot(periodogram, 'Bartlett\'s Method Periodogram - Gaussian Noise Wave')


if __name__ == '__main__':
    print("\nChapter 4: ex_2_bartletts_method.py")
    print("****************************")
    run()
