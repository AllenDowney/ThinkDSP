# The Fourier Transform of a Gaussian curve is also a Gaussian curve. For Discrete Fourier Transforms,
# this relationship is approximately true.
# Try it out for a few examples. What happens to the Fourier Transform as you vary std?

from numpy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt

WINDOW_SIZE = 11


def _plot(signal, title):
    plt.title(title)
    plt.ylabel('Signal Frequency')
    plt.xlabel('Time')
    plt.plot(signal, color='blue')
    plt.show()


def _make_normalized_gaussian_window(std):
    gaussian = signal.windows.gaussian(M=WINDOW_SIZE, std=std, sym=True)
    gaussian /= sum(gaussian)
    return gaussian


def run():
    stds = range(1, 10)
    # M - number of points in the window
    # std - standard deviation
    # sym - flag for a symmetris window or not. Symmetrics is better for filters. 
    # TODO - why does he not use sym = True in the book?
    for std in stds:
        gaussian = _make_normalized_gaussian_window(std)
        fft_gaussian = fft(gaussian)
        _plot(fft_gaussian, f'Gaussian Window Signal std_dev = {std}')


if __name__ == '__main__':
    print("\nChapter 8: ex_2_ft_gaussian.py")
    print("****************************")
    run()

