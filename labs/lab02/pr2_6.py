# Упражнение 2.5

from thinkdsp import SawtoothSignal, CosSignal, ParabolicSignal
from thinkdsp import decorate
import numpy as np
from pr2_5 import filter_spectrum

freq = 500
signal = SawtoothSignal(freq=freq)
wave = signal.make_wave(duration=0.5, framerate=20000)
wave.make_audio()

spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

spectrum.plot(color='gray')
filter_spectrum(spectrum)
spectrum.scale(freq)
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

wave = spectrum.make_wave()
wave.make_audio()
wave.segment(duration=0.01).plot()
decorate(xlabel='Time (s)')

freqs = np.arange(500, 9500, 500)
amps = 1 / freqs**2
signal = sum(CosSignal(freq, amp) for freq, amp in zip(freqs, amps))

spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

wave = signal.make_wave(duration=0.5, framerate=20000)
wave.make_audio()
wave.segment(duration=0.01).plot()
decorate(xlabel='Time (s)')

wave = ParabolicSignal(freq=500).make_wave(duration=0.5, framerate=20000)
wave.make_audio()
wave.segment(duration=0.01).plot()
decorate(xlabel='Time (s)')

spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')