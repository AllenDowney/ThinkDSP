# Using the "Amen" break beat, apply a low-pass filter before sampling, then
# apply the low-pass filter again to remove the spectral copies introduced by sampling.
# The result should be identical to the filtered signal.
import numpy as np

from code.thinkdsp import read_wave, Wave

from exercises.lib.lib import play_wave

SOUND_FILE = 'exercises/ch_11/263868__kevcio__amen-break-a-160-bpm.wav'
FRAMERATE = 22050


def sample(wave, factor=4):
    ys = np.zeros(len(wave))
    ys[::factor] = wave.ys[::factor]
    return Wave(ys, framerate=wave.framerate)


def run():
    source = read_wave(SOUND_FILE)
    print('Plotting source')
    source.plot()
    print('Playing source')
    play_wave(source)

    # Make a brick-wall filter, aka a sinc filter to filter out frequencies above Nyquist
    # which would cause artifacts if source is sampled
    source_spectrum = source.make_spectrum()
    source_spectrum.low_pass(FRAMERATE / 2)
    source = source_spectrum.make_wave()

    print('Plotting anti-aliased source')
    source.plot()
    print('Playing anti-aliased source')
    play_wave(source)

    sampled_source = sample(source)
    print('Plotting sampled source')
    source.plot()
    print('Playing sampled source')
    play_wave(source)


if __name__ == '__main__':
    print("\nChapter 11: ex_11_anti_aliasing.py")
    print("****************************")
    run()
