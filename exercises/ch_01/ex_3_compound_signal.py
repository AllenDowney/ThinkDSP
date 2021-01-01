# Exercise 1-3.  Synthesize a compound signal by creating SinSignal and
# CosSignal objects and adding them up.
# Evaluate the signal to get a Wave, and listen to it.
# Compute its Spectrum and plot it.
# What happens if you add frequency components that are not multiples of the fundamental?

from code.thinkdsp import play_wave, read_wave, SinSignal, CosSignal

TEMP_FILE = 'temp.wav'
# OSX CLI audio player
AUDIO_PLAYER = 'afplay'
FREQ_A4 = 440
FREQ_A5 = 880
FRAMERATE = 22050


def _process_wave(wave):
    # Normalize values to be -1..1
    wave.normalize()
    # Taper envelope at start and end of wave to avoid click when playing/looping
    wave.apodize()


def _play_wave(wave):
    _process_wave(wave)
    wave.write(TEMP_FILE)
    play_wave(TEMP_FILE, player=AUDIO_PLAYER)


def run():
    cos_sig = CosSignal(freq=FREQ_A4, amp=1.0, offset=0)
    sin_sig = SinSignal(freq=FREQ_A5, amp=0.5, offset=0)
    compound_sig = sin_sig + cos_sig

    start = 0.
    duration_secs = 1.
    wave = compound_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing compound sin + cos signal with signal frequencies at A4 and A5')
    _play_wave(wave)

    # Now add a signal that isn't a multiple of the fundamental signal frequency
    dissonant_sig = SinSignal(freq=FREQ_A5 - 25, amp=0.5, offset=0)
    compound_sig += dissonant_sig
    wave = compound_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    print('Playing compound sin + cos signal + dissonant_sig')
    _play_wave(wave)


if __name__ == '__main__':
    run()

