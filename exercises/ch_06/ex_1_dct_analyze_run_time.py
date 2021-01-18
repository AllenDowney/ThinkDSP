# In this chapter I claim that analyze1 takes time proportional to n3 and analyze2
#  takes time proportional to n2. To see if that’s true, run them on a range of
#  input sizes and time them.
# If you plot run time versus input size on a log-log scale, you should get a
#  straight line with slope 3 for analyze1 and slope 2 for analyze2.
# You also might want to test dct_iv and scipy.fftpack.dct.

from time import process_time

import matplotlib.pyplot as plt
import numpy as np

import code.thinkplot as thinkplot

FRAMERATE = 220
TWO_PI = 2.0 * np.pi

def analyze1(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(TWO_PI * args)
    amps = np.linalg.solve(M, ys)
    return amps


def analyze2(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(TWO_PI * args)
    amps = np.dot(M, ys) / 2
    return amps


def synthesize2(amps, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(TWO_PI * args)
    ys = np.dot(M, amps)
    return ys


def _analyze_and_time(N, amps, analyze_func):
    ts = (np.arange(N)) / N
    fs = (np.arange(N)) / 2
    ys = synthesize2(amps, fs, ts)

    start = process_time()
    analyze_func(ys, fs, ts)
    end = process_time()
    return end - start


def _do_run(N, analyze_func, lengths, processing_times):
    # Base value for N is 4, length of amps
    amps = np.array([0.6, 0.25, 0.1, 0.05] * int((N / 4.)))
    processing_time = _analyze_and_time(N, amps, analyze_func)
    lengths.append(N)
    processing_times.append(processing_time)


def _plot(lengths, processing_times, analyze_func_name):
    plt.title(f'Input Length vs. Execution Time - {analyze_func_name}')
    plt.ylabel('Processing Time')
    plt.xlabel('Input Length')
    plt.plot(lengths, processing_times, color='blue')
    plt.show()
    thinkplot.config(xscale='log', yscale='log')


def run():
    lengths = []
    processing_times = []
    N = 4.0
    _do_run(N, analyze1, lengths, processing_times)
    N = 40.0
    _do_run(N, analyze1, lengths, processing_times)
    N = 400.0
    _do_run(N, analyze1, lengths, processing_times)
    N = 4000.0
    _do_run(N, analyze1, lengths, processing_times)
    _plot(lengths, processing_times, 'analyze1')

    lengths = []
    processing_times = []
    N = 4.0
    _do_run(N, analyze2, lengths, processing_times)
    N = 40.0
    _do_run(N, analyze2, lengths, processing_times)
    N = 400.0
    _do_run(N, analyze2, lengths, processing_times)
    N = 4000.0
    _do_run(N, analyze2, lengths, processing_times)
    _plot(lengths, processing_times, 'analyze2')


if __name__ == '__main__':
    print("\nChapter 6: ex_1_dct_analyze_run_time.py")
    print("****************************")
    run()

