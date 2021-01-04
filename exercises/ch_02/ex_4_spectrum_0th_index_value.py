# Make a triangle signal with frequency 440 and make a Wave with duration 0.01 seconds.
#  Plot the waveform.
# Make a Spectrum object and print spectrum.hs[0]. What is the amplitude and phase of
#  this component?
# Set spectrum.hs[0] = 100. What effect does this operation have on the waveform?
#  Hint: Spectrum provides a method called make_wave that computes the Wave that
#  corresponds to the Spectrum.

from code.thinkdsp import TriangleSignal, decorate

FREQ_A4 = 440
FRAMERATE = 22500


def run():
    triangle_sig = TriangleSignal(freq=FREQ_A4, amp=0.5, offset=0)

    start = 0.
    duration_secs = 0.01
    segment = triangle_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)
    segment.plot()
    decorate(xlabel='Time (s)')

    spectrum = segment.make_spectrum()
    print(f'spectrum.hs[0] = {spectrum.hs[0]}')

    spectrum.hs[0] = 100.
    print(f'After changing, spectrum.hs[0] = {spectrum.hs[0]}')
    wave = spectrum.make_wave()
    wave.plot()
    decorate(xlabel='Time (s)')


if __name__ == '__main__':
    print("\nChapter 2: ex_4_spectrum_0th_index_value.py")
    print("****************************")
    run()
