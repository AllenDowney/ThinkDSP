 # The example code in chap05.ipynb shows how to use autocorrelation to estimate the
 #  fundamental frequency of a periodic signal. Encapsulate this code in a function
 #  called estimate_fundamental, and use it to track the pitch of a recorded sound.
 # To see how well it works, try superimposing your pitch estimates on a spectrogram
 #  of the recording.

import numpy as np

from exercises.ch_02.ex_2_sawtooth import SawtoothSignal
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22050


def estimate_fundemental(wave):
    # Compute sequency of correlation coefficients for each pair of segments in the wave
    # Pairs are the wave up to time interval t and from t to the end, for each interval in lags
    correlations = np.correlate(wave.ys, wave.ys, mode='same')
    # Normalize the correlation
    # np.correlate uses the unstandardized definition of correlation; as the lag gets bigger,
    #  the number of points in the overlap between the two signals gets smaller,
    #  so the magnitude of the correlations decreases.
    # We can correct that by dividing through by the lengths
    n = len(correlations)
    lengths = list(range(n, n // 2, -1))
    pos_half = correlations[n // 2:]
    # normalize so correlation with lag == 0 is 1
    pos_half /= pos_half[0]

    # Find first and second peak in the correlations, this is the period of the wave
    # Use first and second because first is arbitrarily offset from the start of the wave
    #  but if the wave is periodic and a high enough frequency to have at least one complete
    #  cycle this algorithm finds that first comlete cycle (the first two peaks). The distance
    #  between them is the distance of the period, i.e. the inverse of the frequency.
    first_peak = None
    second_peak = None
    prev_val = pos_half[0]
    cur_val = 0
    for i in range(len(pos_half) - 1):
        cur_val = pos_half[i]
        next_val = pos_half[i + 1]
        if first_peak is None and prev_val < cur_val < next_val:
            first_peak = cur_val
        elif second_peak is None and prev_val < cur_val < next_val:
            second_peak = cur_val
            break
        prev_val = cur_val
    if first_peak is None or second_peak is None:
        print("Failed to find peaks in correlations. Cannot estimate fundamental")
        return None

    lag_diff = second_peak - first_peak
    period = lag_diff / wave.framerate
    frequency = 1 / period

    return frequency


def run():
    start = 0.
    duration_secs = .01
    sawtooth_sig = SawtoothSignal(freq=FREQ_A4, amp=0.5, offset=0)
    wave = sawtooth_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    fundamental = estimate_fundemental(wave)
    if fundamental is not None:
        print(f'Est. fundamental freq of A440 wave is {(fundamental / 1000):.2f} Hz')


if __name__ == '__main__':
    print("\nChapter 5: ex_2_estimate_fundamental.py")
    print("****************************")
    run()
