# Упражнение 2.3

from thinkdsp import SquareSignal, SinSignal
from thinkdsp import decorate

square = SquareSignal(1100).make_wave(duration=0.5, framerate=10000)
square.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')

square.make_audio()
