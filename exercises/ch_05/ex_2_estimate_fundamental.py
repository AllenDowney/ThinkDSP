 # The example code in chap05.ipynb shows how to use autocorrelation to estimate the
 #  fundamental frequency of a periodic signal. Encapsulate this code in a function
 #  called estimate_fundamental, and use it to track the pitch of a recorded sound.
 # To see how well it works, try superimposing your pitch estimates on a spectrogram
 #  of the recording.

import matplotlib.pyplot as plt
import numpy as np

from exercises.ch_02.ex_2_sawtooth import SawtoothSignal
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22050

def corrcoef(xs, ys):
    """Coefficient of correlation.

    ddof=0 indicates that we should normalize by N, not N-1.

    xs: sequence
    ys: sequence

    returns: float
    """
    return np.corrcoef(xs, ys, ddof=0)[0, 1]


def serial_corr(wave, lag=1):
    """Computes serial correlation with given lag.

    wave: Wave
    lag: integer, how much to shift the wave

    returns: float correlation coefficient
    """
    n = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:n-lag]
    corr = corrcoef(y1, y2)
    return corr


def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    """
    lags = range(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs


def _is_peak(prev_val, cur_val, next_val):
    if any(val <= 0. for val in [prev_val, cur_val, next_val]):
        return False
    return prev_val < cur_val > next_val


def _find_first_period_peak(correlation):
    prev_val = correlation[0]
    cur_val = 0
    for i in range(len(correlation) - 1):
        cur_val = correlation[i]
        next_val = correlation[i + 1]
        if _is_peak(prev_val, cur_val, next_val):
            return i, cur_val
        prev_val = cur_val


def estimate_fundemental(wave):
    lags, corrs = autocorr(wave)
    plt.plot(lags, corrs)
    plt.show()
    lag, _ = _find_first_period_peak(corrs)
    period = lag / wave.framerate
    return 1 / period


def run():
    start = 0.
    duration_secs = .01
    sawtooth_sig = SawtoothSignal(freq=FREQ_A4, amp=0.5, offset=0)
    wave = sawtooth_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    fundamental = estimate_fundemental(wave)
    if fundamental is not None:
        print(f'Est. fundamental freq of A440 wave is {fundamental:.2f} Hz')


if __name__ == '__main__':
    print("\nChapter 5: ex_2_estimate_fundamental.py")
    print("****************************")
    run()
