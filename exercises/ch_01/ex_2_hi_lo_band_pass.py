# Load an audio sample with relatively constant harmonics.
# Use high_pass, low_pass, and band_stop to filter out some of the harmonics
# Then convert the spectrum back to a wave and listen to it.
# How does the sound relate to the changes you made in the spectrum? 

from code.thinkdsp import play_wave, read_wave
from exercises.lib.lib import play_wave

SOUND_FILE = 'exercises/ch_01/SIG_126_A_Retro_Synth.wav'


def run():
    # Assume all exercises run as 'python -m' from project root
    wave = read_wave(SOUND_FILE)

    # Take a segment of the wave
    start = 0.
    duration = 1.5  # secs
    segment = wave.segment(start, duration)

    # Make a spectrum from the segment, and a wave file from that to play
    spectrum = segment.make_spectrum()
    print('Playing unfiltered audio')
    wave = spectrum.make_wave()
    play_wave(wave)

    # High pass
    spectrum = segment.make_spectrum()
    spectrum.high_pass(cutoff=1000, factor=0.01)
    print('Playing audio with high-pass filter')
    wave = spectrum.make_wave()
    play_wave(wave)

    # Low pass
    spectrum = segment.make_spectrum()
    spectrum.low_pass(cutoff=1000, factor=0.01)
    print('Playing audio with low-pass filter')
    wave = spectrum.make_wave()
    play_wave(wave)

    # Band pass
    spectrum = segment.make_spectrum()
    spectrum.band_stop(600, 1000, factor=0.01)
    print('Playing audio with band-pass filter')
    wave = spectrum.make_wave()
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 1: ex_2_hi_lo_band_pass.py")
    print("****************************")
    run()
