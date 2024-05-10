# Упражнение 4.1

from thinkdsp import read_wave, decorate

wave = read_wave('../../code/132736__ciccarelli__ocean-waves.wav')
wave.make_audio()

segment = wave.segment(start=1.5, duration=1.0)
segment.make_audio()

spectrum = segment.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)', ylabel='Power')

spectrum.plot_power()

log_log = dict(xscale='log', yscale='log')
decorate(xlabel='Frequency (Hz)', ylabel='Power', **log_log)

segment2 = wave.segment(start=2.5, duration=1.0)
segment2.make_audio()

spectrum2 = segment2.make_spectrum()

spectrum.plot_power(alpha=0.5)
spectrum2.plot_power(alpha=0.5)
decorate(xlabel='Frequency (Hz)', ylabel='Power')

spectrum.plot_power(alpha=0.5)
spectrum2.plot_power(alpha=0.5)
decorate(xlabel='Frequency (Hz)', ylabel='Power', **log_log)

segment.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time(s)', ylabel='Frequency (Hz)')
