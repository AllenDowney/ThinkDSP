# Упражнение 4.5

import pandas as pd
from pprint import pprint
import numpy as np
from thinkdsp import Wave
from thinkdsp import decorate
from pr4_1 import log_log
from pr4_2 import bartlett_method


def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm."""

    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


nrows = 100
ncols = 5

array = np.empty((nrows, ncols))
array.fill(np.nan)
array[0, :] = np.random.random(ncols)
array[:, 0] = np.random.random(nrows)
pprint(array[0:6])

p = 0.5
n = nrows
cols = np.random.geometric(p, n)
cols[cols >= ncols] = 0
pprint(cols)

rows = np.random.randint(nrows, size=n)
pprint(rows)

array[rows, cols] = np.random.random(n)
pprint(array[0:6])

df = pd.DataFrame(array)
df.head()

filled = df.fillna(method='ffill', axis=0)
filled.head()

total = filled.sum(axis=1)
total.head()

wave = Wave(total.values)
wave.plot()

ys = voss(11025)
pprint(ys)

wave = Wave(ys)
wave.unbias()
wave.normalize()
wave.plot()

wave.make_audio()
spectrum = wave.make_spectrum()
spectrum.hs[0] = 0
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)', ylabel='Power', **log_log)

print(spectrum.estimate_slope().slope)

seg_length = 64 * 1024
iters = 100
wave = Wave(voss(seg_length * iters))
print(len(wave))

spectrum = bartlett_method(wave, seg_length=seg_length, win_flag=False)
spectrum.hs[0] = 0
print(len(spectrum))

spectrum.plot_power()
decorate(xlabel='Frequency (Hz)', ylabel='Power', **log_log)

print(spectrum.estimate_slope().slope)
