# Упражнение 1.2

from thinkdsp import read_wave

# Достаём волну из файла, проигрываем её и строим график
wave = read_wave('../../code/170255__dublie__trumpet.wav')
wave.normalize()
wave.make_audio()
wave.plot()

# Вырезаем сегмент с постоянной высотой из волны и строим график
segment = wave.segment(start=1.1, duration=0.3)
segment.make_audio()
segment.plot()

# Вырезаем сегмент ещё меньше и строим график
smaller_seg = segment.segment(start=1.1, duration=0.005)
smaller_seg.plot()

# Построим спектр
spectrum = segment.make_spectrum()
spectrum.plot(high=7000)

# Отсекаем частоты выше 1000
spectrum = segment.make_spectrum()
spectrum.plot(high=1000)

# Распечатаем высшие точки спектра и их частоты в порядке убывания
from pprint import pprint
pprint(spectrum.peaks()[:30])
print()

# Отфильтруем высокие частоты
spectrum.low_pass(2000)
spectrum.make_wave().make_audio()

# Отфильтруем низкие частоты (звучит ужасно)
spectrum.high_pass(2000)
spectrum.make_wave().make_audio()

# Отфильтруем компоненты в полосе частот между двумя частотами среза
spectrum.band_stop(2000, 5000)
spectrum.make_wave().make_audio()