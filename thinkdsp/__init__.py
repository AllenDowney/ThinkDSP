"""ThinkDSP: Digital Signal Processing in Python.

This module provides classes and functions for working with signals,
waves, and spectrums, as described in "Think DSP" by Allen B. Downey.
"""

from .thinkdsp import (
    # Constants
    PI2,
    # Exceptions
    UnimplementedMethodException,
    # Utility functions
    random_seed,
    find_index,
    unbias,
    normalize,
    shift_right,
    shift_left,
    truncate,
    quantize,
    apodize,
    zero_pad,
    mag,
    infer_framerate,
    # Wave I/O functions
    read_wave,
    read_wave_with_scipy,
    play_wave,
    # Signal classes
    Signal,
    SumSignal,
    Sinusoid,
    CosSignal,
    SinSignal,
    Sinc,
    ComplexSinusoid,
    SquareSignal,
    SawtoothSignal,
    ParabolicSignal,
    CubicSignal,
    GlottalSignal,
    TriangleSignal,
    Chirp,
    ExpoChirp,
    SilentSignal,
    Impulses,
    Noise,
    UncorrelatedUniformNoise,
    UncorrelatedGaussianNoise,
    BrownianNoise,
    PinkNoise,
    # Wave and spectrum classes
    Wave,
    WavFileWriter,
    Spectrum,
    IntegratedSpectrum,
    Dct,
    Spectrogram,
    # Helper functions for creating signals
    make_note,
    make_chord,
    midi_to_freq,
    sin_wave,
    cos_wave,
    rest,
    # Plotting utilities
    decorate,
    legend,
    remove_from_legend,
    underride,
)

# Define __all__ for explicit exports
__all__ = [
    # Constants
    "PI2",
    # Exceptions
    "UnimplementedMethodException",
    # Utility functions
    "random_seed",
    "find_index",
    "unbias",
    "normalize",
    "shift_right",
    "shift_left",
    "truncate",
    "quantize",
    "apodize",
    "zero_pad",
    "mag",
    "infer_framerate",
    # Wave I/O functions
    "read_wave",
    "read_wave_with_scipy",
    "play_wave",
    # Signal classes
    "Signal",
    "SumSignal",
    "Sinusoid",
    "CosSignal",
    "SinSignal",
    "Sinc",
    "ComplexSinusoid",
    "SquareSignal",
    "SawtoothSignal",
    "ParabolicSignal",
    "CubicSignal",
    "GlottalSignal",
    "TriangleSignal",
    "Chirp",
    "ExpoChirp",
    "SilentSignal",
    "Impulses",
    "Noise",
    "UncorrelatedUniformNoise",
    "UncorrelatedGaussianNoise",
    "BrownianNoise",
    "PinkNoise",
    # Wave and spectrum classes
    "Wave",
    "WavFileWriter",
    "Spectrum",
    "IntegratedSpectrum",
    "Dct",
    "Spectrogram",
    # Helper functions for creating signals
    "make_note",
    "make_chord",
    "midi_to_freq",
    "sin_wave",
    "cos_wave",
    "rest",
    # Plotting utilities
    "decorate",
    "legend",
    "remove_from_legend",
    "underride",
]

__version__ = "0.1.1"
