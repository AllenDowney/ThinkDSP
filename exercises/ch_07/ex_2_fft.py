# Given a wave array, y, split it into its even elements, e, and its odd elements, o
# Compute the DFT of e and o by making recursive call
# Compute DFT(y) for each value of n using the Danielson–Lanczos lemma

import numpy as np

from code.thinkdsp import PI2, SawtoothSignal

FRAMERATE = 10000
AMPS = np.array([0.6, 0.25, 0.1, 0.05])
FS = [100, 200, 300, 400]

EVEN = 0
ODD = 1


def _split_wave(wave):
    if len(wave.ys) % 2:
        raise ValueError('Invalid wave array, length must be even')
    evens = np.copy(wave.ys)[::2]
    odds = np.copy(wave.ys)[1::2]
    return evens, odds


# split - a tuple of (e, o)
# e - even index values of original wave
# o - odd index values of original wave
# ne_no - list of current indexes into processing e and o, first index is e and second is o
#  list because it gets updated
# even_odd - a const that serves both to identify the input being processed on this step
#  and as the offset into split to access it
# N - total length of e and o combined, i.e. length of original wave
def _fft(split, even_odd, ne_no, N, accum):
    if ne_no[even_odd] == 0:
        next_fft = split[even_odd][ne_no[even_odd]]
        accum.append(next_fft)
        return next_fft
    else:
        curr_n = ne_no[even_odd]
        ne_no[even_odd] = ne_no[even_odd] - 1
        next_fft = _fft(split, EVEN, ne_no, N, accum) + (_fft(split, ODD, ne_no, N, accum) * \
            (np.exp((PI2 * 1j * curr_n) / N)))
        accum.append(next_fft)
        return next_fft

def run():
    freq = 500
    duration = 0.1
    signal = SawtoothSignal(freq=freq)
    wave = signal.make_wave(duration=duration, framerate=FRAMERATE)
    e, o = _split_wave(wave)
    accum = []
    _fft((e, o), EVEN, [len(e), len(o)], len(wave.ys), accum)
    print(accum)


if __name__ == '__main__':
    print("\nChapter 7: ex_2_fft.py")
    print("****************************")
    run()
