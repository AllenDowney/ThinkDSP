from code.thinkdsp import play_wave as dsp_play_wave

TEMP_FILE = 'temp.wav'
# OSX CLI audio player
AUDIO_PLAYER = 'afplay'


def _process_wave(wave):
    # Normalize values to be -1..1
    wave.normalize()
    # Taper envelope at start and end of wave to avoid click when playing/looping
    wave.apodize()


def play_wave(wave):
    _process_wave(wave)
    wave.write(TEMP_FILE)
    dsp_play_wave(TEMP_FILE, player=AUDIO_PLAYER)

