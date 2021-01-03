from code.thinkdsp import TriangleSignal, decorate
from exercises.lib.lib import play_wave

FREQ_A4 = 440
FRAMERATE = 22500


def run():
    triangle_sig = TriangleSignal(freq=FREQ_A4, amp=0.5, offset=0)

    start = 0.
    duration_secs = 0.01
    segment = triangle_sig.make_wave(start=0, duration=duration_secs, framerate=FRAMERATE)

    # import pdb; pdb.set_trace()

    segment.plot()
    decorate(xlabel='Time (s)')

    # print('Playing square signal without aliasing')
    # play_wave(wave)



if __name__ == '__main__':
    print("\nChapter 2: ex_4_spectrum_0th_index_value.py")
    print("****************************")
    run()
