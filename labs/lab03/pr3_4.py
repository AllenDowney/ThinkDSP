# Упражнение 3.4

from thinkdsp import read_wave, decorate

wave = read_wave('../../code/72475__rockwehrmann__glissup02.wav')
wave.make_audio()

wave.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')
