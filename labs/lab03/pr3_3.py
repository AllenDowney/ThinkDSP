# Упражнение 3.3

from thinkdsp import decorate
from pr3_2 import SawtoothChirp

signal = SawtoothChirp(start=2500, end=3000)
wave = signal.make_wave(duration=1, framerate=20000)
wave.make_audio()

wave.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')
