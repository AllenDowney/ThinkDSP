# Load an audio sample with relatively constant harmonics.
# Use high_pass, low_pass, and band_stop to filter out some of the harmonics
# Then convert the spectrum back to a wave and listen to it.
# How does the sound relate to the changes you made in the spectrum? 

from code.thinkdsp import play_wave, read_wave


SOUND_FILE = 'exercises/ch_01/SIG_126_A_Retro_Synth.wav'
TEMP_FILE = 'temp.wav'
# OSX CLI audio player
AUDIO_PLAYER = 'afplay'


def _process_wave(wave):
    # Normalize values to be -1..1
    wave.normalize()
    # Taper envelope at start and end of wave to avoid click when playing/looping
    wave.apodize()


def _play_spectrum(spectrum):
    wave = spectrum.make_wave()
    _process_wave(wave)
    wave.write(TEMP_FILE)
    play_wave(TEMP_FILE, player=AUDIO_PLAYER)


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
    _play_spectrum(spectrum)

    # High pass
    spectrum = segment.make_spectrum()
    spectrum.high_pass(cutoff=1000, factor=0.01)
    print('Playing audio with high-pass filter')
    _play_spectrum(spectrum)

    # Low pass
    spectrum = segment.make_spectrum()
    spectrum.low_pass(cutoff=1000, factor=0.01)
    print('Playing audio with low-pass filter')
    _play_spectrum(spectrum)

    # Band pass
    spectrum = segment.make_spectrum()
    spectrum.band_stop(600, 1000, factor=0.01)
    print('Playing audio with band-pass filter')
    _play_spectrum(spectrum)


if __name__ == '__main__':
    run()
