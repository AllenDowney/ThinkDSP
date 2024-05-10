# Упражнение 4.3

import pandas as pd
from thinkdsp import Wave
from thinkdsp import decorate
from pr4_1 import log_log

df = pd.read_csv(
    filepath_or_buffer='BTC_USD_2013-10-01_2020-03-26-CoinDesk.csv',
    parse_dates=[0]
)

ys = df['Closing Price (USD)']
ts = df.index

wave = Wave(ys, ts, framerate=1)
wave.plot()
decorate(xlabel='Time (days)')

spectrum = wave.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (1/days)', ylabel='Power', **log_log)

print(spectrum.estimate_slope()[0])
