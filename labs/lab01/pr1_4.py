# Упражнение 1.4

from thinkdsp import read_wave


def stretch(wave, factor):
    wave.ts *= factor
    wave.framerate /= factor


wave = read_wave('../../code/170255__dublie__trumpet.wav')
wave.normalize()
wave.make_audio()

stretch(wave, 0.5)  # ускоряем запись в два раза
wave.make_audio()

wave.plot()
