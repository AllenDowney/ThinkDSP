 # If you did the exercises in the previous chapter, you downloaded the historical prices
 # of BitCoins and estimated the power spectrum of the price changes. Using the same
 # data, compute the autocorrelation of BitCoin prices. Does the autocorrelation
 # function drop off quickly? Is there evidence of periodic behavior?

import csv
from math import floor

import matplotlib.pyplot as plt
import numpy as np

DATA = 'exercises/ch_04/BTC_USD_2020-01-10_2021-01-09-CoinDesk.csv'

def corrcoef(xs, ys):
    """Coefficient of correlation.

    ddof=0 indicates that we should normalize by N, not N-1.

    xs: sequence
    ys: sequence

    returns: float
    """
    return np.corrcoef(xs, ys, ddof=0)[0, 1]


def serial_corr(ys, lag=1):
    """Computes serial correlation with given lag.

    ys: sequence of floats
    lag: integer, how much to shift the wave

    returns: float correlation coefficient
    """
    n = len(ys)
    y1 = ys[lag:]
    y2 = ys[:n-lag]
    corr = corrcoef(y1, y2)
    return corr


def autocorr(ys):
    """Computes and plots the autocorrelation function.

    ys: sequence of floats
    """
    lags = range(len(ys)//2)
    corrs = [serial_corr(ys, lag) for lag in lags]
    return lags, corrs



def _get_data():
    price_data = None
    with open(DATA) as f:
        data_reader = csv.DictReader(f, delimiter=',')
        price_data = [[row['Date'], row['Closing Price (USD)']] for row in data_reader]
    dates = [pd[0] for pd in price_data]
    # Convert to int because matplotlib not plotting floats. I probably just don't know
    # how to use it yet.
    prices = [int(floor(float(pd[1]))) for pd in price_data]
    return dates, prices


def _plot_data(dates, prices, title, xlabel, ylabel):
    _, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.plot(list(range(len(dates))), prices, color='blue')
    plt.show()


def run():
    dates, prices = _get_data()
    title = 'Bitcoin (BTC) Closing Price by day - 2020'
    xlabel = f'Dates: {dates[0]} - {dates[-1]}'
    ylabel = 'Price (USD)'
    _plot_data(dates, prices, title, xlabel, ylabel)

    lags, corrs = autocorr(prices)
    title = 'Bitcoin (BTC) Autocorrelation by Day - 2020'
    xlabel = f'Interval in year: {lags[0]} - {lags[-1]}'
    ylabel = 'Autocorrelation'
    _plot_data(lags, corrs, title, xlabel, ylabel)


if __name__ == '__main__':
    print("\nChapter 5: ex_3_bitcoin_autocorrelation.py")
    print("****************************")
    run()

