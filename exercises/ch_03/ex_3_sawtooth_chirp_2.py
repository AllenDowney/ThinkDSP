 # Make a sawtooth chirp that sweeps from 2500 to 3000 Hz, then use it to make a wave with
 # duration 1 s and frame rate 20 kHz. Draw a sketch of what you think the spectrum will
 # look like. Then plot the spectrum and see if you got it right. 

from exercises.ch_03.ex_2_sawtooth_chirp import SawtoothChirpSignal
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 20
CHIRP_START_FREQ = 2500
CHIRP_END_FREQ = 3000
OFFSET = 0
START = 0
DURATION_SECS = 1
AMP = 0.5


def run():
    sawtooth_chirp_sig = SawtoothChirpSignal(CHIRP_START_FREQ, CHIRP_END_FREQ, FREQ_A4,
                                             framerate=FRAMERATE)
    # Note this actually just plots three periods of the wave
    wave = sawtooth_chirp_sig.make_wave(start=START, duration=DURATION_SECS, framerate=FRAMERATE)
    print('Plotting sawtooth chirp wave with very low framerate. Verify the waveform.')
    wave.plot()
    # Not enough samples to generate audible output.
    print('Playing sawtooth chirp wave with very low framerate. Expect no audio.')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 3: ex_2_sawtooth_chirp.py")
    print("****************************")
    run()

