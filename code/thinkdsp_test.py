"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import unittest
import thinkdsp

import numpy as np

class Test(unittest.TestCase):

    def testComplexSinusoid(self):
        signal = thinkdsp.ComplexSinusoid(440, 0.7, 1.1)
        result = signal.evaluate(2.1) * complex(-1.5, -0.5)

        self.assertAlmostEqual(result, -0.164353351475-1.09452637056j)

    def testWaveAdd(self):
        ys = np.array([1, 2, 3, 4])
        wave1 = thinkdsp.Wave(ys, framerate=1)
        wave2 = wave1.copy()
        wave2.shift(2)
        wave = wave1 + wave2

        self.assertEqual(len(wave), 6)
        self.assertAlmostEqual(sum(wave.ys), 20)

    def testDct(self):
        signal = thinkdsp.CosSignal(freq=2)
        wave = signal.make_wave(duration=1, framerate=8)
        dct = wave.make_dct()
        
        self.assertAlmostEqual(dct.fs[0], 0.25)

    def testImpulses(self):
        imp_sig = thinkdsp.Impulses([0.01, 0.4, 0.8, 1.2], 
                                    amps=[1, 0.5, 0.25, 0.1])
        impulses = imp_sig.make_wave(start=0, duration=1.3, 
                                     framerate=11025)

        self.assertAlmostEqual(len(impulses), 14333)


if __name__ == "__main__":
    unittest.main()
