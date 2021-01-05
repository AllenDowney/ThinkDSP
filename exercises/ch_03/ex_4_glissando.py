 # In musical terminology, a “glissando” is a note that slides from one pitch to another,
 #  so it is similar to a chirp.
 # Find or make a recording of a glissando and plot a spectrogram of the first few seconds.
 # One suggestion: George Gershwin’s Rhapsody in Blue, which you can download
 #  from http://archive.org/details/rhapblue11924, starts with a famous clarinet glissando.

from code.thinkdsp import read_wave
from exercises.lib.lib import play_wave

SOUND_FILE = 'exercises/ch_03/gershwin_rhapsody_in_blue.wav'


def run():
    wave = read_wave(SOUND_FILE)
    print('Plotting glissando from "Rhapshody in Blue"')
    wave.plot()
    print('Playing glissando from "Rhapshody in Blue"')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 3: ex_4_glissando.py")
    print("****************************")
    run()
