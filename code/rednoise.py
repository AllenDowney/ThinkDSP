"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy
import matplotlib.pyplot as pyplot
import scipy.stats
import scipy.integrate

duration = 10.0
framerate = 11025
dx = 1.0 / framerate
n = framerate * duration
print n

dys = numpy.random.uniform(-1, 1, n)
ys = scipy.integrate.cumtrapz(dys, dx=dx)
#ys = scipy.integrate.simps(dys)
#ys = numpy.cumsum(dys)
ys -= numpy.mean(ys)

hs = numpy.fft.rfft(ys)

m = len(hs)
max_freq = framerate / 2.0
fs = numpy.linspace(0, max_freq, m)
ps = numpy.absolute(hs) ** 2

x = numpy.log(fs[1:])
y = numpy.log(ps[1:])
slope, _, _, _, _ = scipy.stats.linregress(x,y)
print slope

pyplot.plot(fs[1:], ps[1:])
pyplot.xscale('log')
pyplot.yscale('log')
#pyplot.show()

