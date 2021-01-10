#  At http://www.coindesk.com/price, you can download historical data on the
# daily price of a BitCoin as a CSV file. Read this file and compute the spectrum of
# BitCoin prices as a function of time. Does it resemble white, pink, or Brownian noise?

import csv
from math import floor

import matplotlib.pyplot as plt

DATA = 'exercises/ch_04/BTC_USD_2020-01-10_2021-01-09-CoinDesk.csv'


def run():
    price_data = None
    with open(DATA) as f:
        data_reader = csv.DictReader(f, delimiter=',')
        price_data = [[row['Date'], row['Closing Price (USD)']] for row in data_reader]
    dates = [pd[0] for pd in price_data]
    # Convert to int because matplotlib not plotting floats. I probably just don't know
    # how to use it yet.
    prices = [int(floor(float(pd[1]))) for pd in price_data]
    _, axes = plt.subplots()
    axes.set_title('Bitcoin (BTC) Closing Price by day - 2020')
    axes.set_xlabel(f'Dates: {dates[0]} - {dates[-1]}')
    axes.set_ylabel('Price (USD)')
    axes.plot(list(range(len(dates))), prices, color='blue')
    plt.show()


if __name__ == '__main__':
    print("\nChapter 4: ex_3_bitcoin_price.py")
    print("****************************")
    run()
