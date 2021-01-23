# One of the major applications of the DCT is compression for both sound and images.
# In its simplest form, DCT-based compression works like this:
# * Break a long signal into segments.
# * Compute the DCT of each segment.
# * Identify frequency components with amplitudes so low they are inaudible, and remove them. Store only the frequencies and amplitudes that remain.
# * To play back the signal, load the frequencies and amplitudes for each segment and apply the inverse DCT. 
# Implement a version of this algorithm and apply it to a recording of music or
#  speech. How many components can you eliminate before the difference is perceptible? 


# TODO CURRENT IMPL. IS WRONG
# "Now we are ready to solve the analysis problem. Suppose I give you a wave and
# tell you that it is the sum of cosines with a given set of frequencies.
# How would you find the amplitude for each frequency component? In other words,
# given ys, ts, and fs, can you recover amps?"
#
# We need to call analyze() for each segment, get back an array of amps which
#  are the amps for the frequencies in fs a the same ordinal positions. Then we
#  we need to set the amps to 0 for all positions which are below threshold. Then
#  we need to make a wave from that ys (which we just modified, the one returned
#  from analyze() (DCT), and the fs which was passed to analyze, which has freqs
#  at the matching indexes. Effectively then we synthesize a new wave from the same
#  fs with new amps which are selectively set to 0, thus eliminating those
#  low-amp frequencies from the new wave.

# To do the above we need to realize the amps from analyze() aren't amps for a
#  a wave, they are *weights* for the frequency array given to analyze. Which is
#  you pass them to inverse_dct() to get the wave from the DCT, because that is
#  what that function does, synthesizes a wave from a DCT. Also, we can't do this
#  for a sample, because we can't control how many initial amp weights there are.
# So we must use synthesize() like in the book chapter, passing in a known array
#  of initial amps, which weight each one, and then taking a subset of that length
#  for fs and ts when we pass that to analyze()


import matplotlib.pyplot as plt
import numpy as np

from code.thinkdsp import read_wave, CosSignal, SumSignal, Wave
import code.thinkplot as thinkplot

from exercises.lib.lib import play_wave
from exercises.ch_06.ex_1_dct_analyze_run_time import analyze1 as analyze

FRAMERATE = 11025
AMPS = np.array([0.6, 0.25, 0.1, 0.05])
FS = [100, 200, 300, 400]


# Create frequencies from a number of Cosin components of know amps, fs, ts
def synthesize(amps, fs, ts):
    components = [CosSignal(freq, amp)
                  for amp, freq in zip(amps, fs)]
    signal = SumSignal(*components)
    ys = signal.evaluate(ts)
    return ys


def _combine_segments(segments):
    combined_ys = np.array([])
    combined_ts = np.array([])
    for segment in segments:
        combined_ys = np.append(combined_ys, segment.ys)
        combined_ts = np.append(combined_ts, segment.ts)
    return Wave(combined_ys, ts=combined_ts, framerate=FRAMERATE)


def _play_compressed(segment_dcts, ts, compression_factor):
    print(f'Playing section segments dropping amps <= {compression_factor * 100}% of mean')
    segments = []
    for amps in segment_dcts:
        threshold = (amps / len(amps)) * compression_factor
        amps[amps <= threshold] = 0.
        ys = synthesize(amps, FS, ts)
        segment = Wave(ys, FRAMERATE)
        segments.append(segment)
    wave = _combine_segments(segments)
    play_wave(wave)


def run():
    ts = np.linspace(0, 1, FRAMERATE)
    ys = synthesize(AMPS, FS, ts)
    wave = Wave(ys, ts=ts, framerate=FRAMERATE)

    n = len(FS)
    start = 0.
    duration_secs = 0.1
    num_segments = int(1. / duration_secs)
    segment_dcts = []
    for i in range(num_segments):
        segment = wave.segment(start, duration_secs)
        spectrum = segment.make_spectrum()
        # TODO THIS IS A BUG. len of spectrum.fs is rounded 1/2 the length of the
        #  the segment frequencies and times. This truncates the segment but that is
        #  almost certainly wrong. It's probably more accurate to interpolate the shorter
        #  sequence into the longer and use every other value for ys and ts, but not sure.
        segment_dcts.append(analyze(segment.ys[:len(spectrum.fs)],
                                    spectrum.fs,
                                    segment.ts[:len(spectrum.fs)]))
        start += duration_secs

    print('Playing uncompressed full source segment')
    play_wave(wave)
    # play_wave(wave.segment(0, num_segments * duration_secs))

    compression_factor = .00001
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .0001
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .001
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .01
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .05
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .1
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .15
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .2
    _play_compressed(segment_dcts, ts, compression_factor)

    compression_factor = .5
    _play_compressed(segment_dcts, ts, compression_factor)


if __name__ == '__main__':
    print("\nChapter 6: ex_2_compression.py")
    print("****************************")
    run()
