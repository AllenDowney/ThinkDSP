from unum.units import OHM, F


r = 10.0 / 255 * 1000 * OHM
c = 4.7 * 1e-6 * F


def frequency(n, m, p):
    num = 1.0/n + 1.0 / (p+m)
    den = r * c
    return num / den

m = 0
p = 256

freqs = []
for n in range(1, 256):
    for m in range(0, 256):
        f = frequency(n, m, p)
        freqs.append(f)

freqs.sort()
print freqs[0], freqs[-1]
print len(freqs)
