# Упражнение 3.6

from thinkdsp import read_wave, decorate

wave = read_wave('../../code/87778__marcgascon7__vocals.wav')
wave.make_audio()

wave.make_spectrogram(1024).plot(high=1000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

high = 1000

segment = wave.segment(start=1, duration=0.25)    # ah
segment.make_spectrum().plot(high=high)

segment = wave.segment(start=2.2, duration=0.25)  # eh
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

segment = wave.segment(start=3.5, duration=0.25)  # ih
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

segment = wave.segment(start=5.1, duration=0.25)  # oh
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

segment = wave.segment(start=6.5, duration=0.25)  # oo
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')
