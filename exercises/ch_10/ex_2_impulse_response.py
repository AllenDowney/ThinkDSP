# Simulate the sound of your recording in the space where the impulse response was measured,
# computed two ways: by convolving the recording with the impulse response and by computing
# the filter that corresponds to the impulse response and multiplying by the DFT of the
# recording.
import numpy as np

from code.thinkdsp import read_wave, Wave

from exercises.lib.lib import play_wave

SOUND_FILE = 'exercises/ch_01/SIG_126_A_Retro_Synth.wav'
IMPULSE_RESPONSE_FILE = 'exercises/ch_10/7_r_churchmouth_s_churchpath.wav'
FRAMERATE = 22050


def run():
    impulse_response = read_wave(IMPULSE_RESPONSE_FILE)
    # Trim silence from front of file
    impulse_response = impulse_response.segment(start=0.042, duration=2.)
    impulse_response.normalize()
    print('Plotting impulse response')
    impulse_response.plot()
    print('Playing impulse response')
    play_wave(impulse_response)

    source = read_wave(SOUND_FILE)
    # Trim to be the same length as the impulse response, offset a bit from the beginning
    # just to avoid any artifacts from start of sample. truncate() matches length exactly.
    source.truncate(len(impulse_response))
    source.normalize()
    print('Plotting source')
    source.plot()
    print('Playing source before convolution') 
    play_wave(source)

    # Convert impulse response to transfer function, that is take its spectrum, that is
    # convert from time domain to frequency domain
    transfer_function = impulse_response.make_spectrum()
    print('Plotting transfer function (spectrum of impulse response)')
    transfer_function.plot()
    # Take DFT of source
    source_spectrum = source.make_spectrum()
    # Compute product of the spectrums, which is product of the DFT of source and impulse.
    # This is modifies the source by the system (is the equivalent of convolution).
    output = (source_spectrum * transfer_function).make_wave()
    print('Plotting wave of source modified by transfer function')
    output.plot()
    output.normalize()
    print('Playing wave of source modified by transfer function')
    play_wave(output)

    # We can perform the same operation as a convolution in the time domain, over the source
    # signal as a wave and with the impulse wave as the convolution window
    window = impulse_response.ys
    convolved = np.convolve(source.ys, window)
    convolved_output = Wave(convolved, framerate=FRAMERATE)
    print('Plotting wave of convolved output')
    convolved_output.plot()
    convolved_output.normalize()
    print('Playing wave of convolved output')
    play_wave(convolved_output)


if __name__ == '__main__':
    print("\nChapter 10: ex_2_impulse_response.py")
    print("****************************")
    run()
