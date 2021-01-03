# Make a square signal at 1100 Hz and make a Wave that samples it at 10,000 frames per second.
# If you plot the spectrum, you can see that most of the harmonics are aliased.
# When you listen to the wave, can you hear the aliased harmonics?

from code.thinkdsp import read_wave, SquareSignal
from exercises.lib.lib import play_wave

SIGNAL_FREQ = 1100.
NYQUIST_FREQ = SIGNAL_FREQ * 2.


def run():
    square_sig = SquareSignal(freq=SIGNAL_FREQ, amp=0.5, offset=0)

    start = 0.
    duration_secs = 1.
    wave = square_sig.make_wave(start=0, duration=duration_secs, framerate=NYQUIST_FREQ)
    print('Playing square signal without aliasing')
    play_wave(wave)

    frequency_exceeding_nyquist = NYQUIST_FREQ * 2.
    wave = square_sig.make_wave(start=0, duration=duration_secs,
                                framerate=frequency_exceeding_nyquist)
    print(f'Playing square signal with aliasing, frequency = {frequency_exceeding_nyquist}')
    play_wave(wave)

    frequency_exceeding_nyquist = 10000.
    wave = square_sig.make_wave(start=0, duration=duration_secs,
                                framerate=frequency_exceeding_nyquist)
    print(f'Playing square signal with aliasing, frequency = {frequency_exceeding_nyquist}')
    play_wave(wave)


if __name__ == '__main__':
    print("\nChapter 2: ex_3_square_aliased.py")
    print("****************************")
    run()

