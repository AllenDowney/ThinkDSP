# Упражнение 1.3

from thinkdsp import SinSignal

signal = (SinSignal(freq=400, amp=1.0) +
          SinSignal(freq=600, amp=0.5) +
          SinSignal(freq=800, amp=0.25))
signal.plot()

wave = signal.make_wave(duration=1)
wave.apodize()
wave.make_audio()

spectrum = wave.make_spectrum()
spectrum.plot(high=2000)

signal += SinSignal(freq=450)
signal.make_wave().make_audio()
