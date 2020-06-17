import numpy as np

import matplotlib.pyplot as plt

def plot_orderbook():
    batches = np.load('/home/ralph/dev/smartlab/data/snapshots/depth-100/BTC-USD_test/2020-04-05_0.npy')
    for sn_ind in range(1000):
        plt.clf()
        bid_prices = batches[sn_ind][0, :, 0].reshape(100)
        ask_prices = batches[sn_ind][1, :, 0].reshape(100)

        bid_sizes = batches[sn_ind][0, :, 1].reshape(100)
        ask_sizes = batches[sn_ind][1, :, 1].reshape(100)

        plt.plot(bid_prices, bid_sizes, 'ro')
        plt.plot(ask_prices, ask_sizes, 'bo')
        plt.show()

def average(x):
    return (x[0][0][0] + x[1][0][0]) / 2

def calculate_labels_mid_prices(current_snapshots, labels_lookahead):
    mid_prices = []
    for i in range(0, len(current_snapshots)):
        mid_prices.append(average(current_snapshots[i]))

    mid_prices = np.array(mid_prices)
    labels = []

    for i in range(0, len(current_snapshots) - labels_lookahead):
        next_midprices = mid_prices[i + 1: i + labels_lookahead]
        next_midprices_avg = np.sum(next_midprices) / len(next_midprices)
        labels.append(mid_prices[i] < next_midprices_avg)
    return mid_prices, labels


def plot_mid_prices(snapshots, plot_title, size):
    snapshots = snapshots[400:700]
    mid_prices, labels = calculate_labels_mid_prices(snapshots, 10)

    plt.plot(mid_prices)

    positive_ind = [idx for idx, asd in enumerate(labels) if asd == 1]
    # natural_ind = [idx for idx, asd in enumerate(labels) if asd == 1]
    negative_ind = [idx for idx, asd in enumerate(labels) if asd == 0]

    # greens = plt.scatter(positive_ind, mid_prices[positive_ind], color='green', s=size)
    # # cyans = plt.scatter(natural_ind, prices[natural_ind], color='cyan', s=size)
    # reds = plt.scatter(negative_ind, mid_prices[negative_ind], color='red', s=size)

    for i in range(0, len(snapshots)):
        # plt.axhspan(i, i + .2, facecolor='0.2', alpha=0.5)
        if i in positive_ind:
            plt.axvspan(i, i + 1, facecolor='g', alpha=0.5)
        if i in negative_ind:
            plt.axvspan(i, i + 1, facecolor='r', alpha=0.5)
    plt.xlabel('Number of snapshot')
    plt.ylabel('Middle price (USD)')
    # plt.legend((greens, reds),
    #            ('Positive sample', 'Negative sample'),
    #            loc='lower left')
    plt.title(plot_title)
    plt.savefig('./doc-figs/price-example.png')
    plt.show()


def plot_snapshot(snapshots):
        snapshot = snapshots[0]
        bid_prices = snapshot[0][:100,0]
        bid_sizes = snapshot[0][:100,1]

        ask_prices = snapshot[1][:100,0]
        ask_sizes = snapshot[1][:100,1]

        plt.bar(bid_prices, bid_sizes)
        plt.bar(ask_prices, ask_sizes)
        plt.show()

import matplotlib.patches as mpatches

def plot_snapshot(snapshots):
    plt.clf()
    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.set_ylim([0, 200])

    plt.xlabel('Available amount (BTC)')
    plt.ylabel('Price Level (USD)')
    plt.title('Cumulated view of the BTC-USD order book')

    snapshot = snapshots[0]
    bid_prices = snapshot[0][:100, 0]
    bid_sizes = np.cumsum(snapshot[0][:100, 1])

    ask_prices = snapshot[1][:100, 0]
    ask_sizes = np.cumsum(snapshot[1][:100, 1])

    print('Drawing line', str(bid_prices[0] - 10), str(bid_sizes[0 + 1]), str(bid_sizes[0]))
    for i in range(0, len(bid_prices) - 1):
        plt.vlines(bid_prices[i + 1], bid_sizes[i + 1], bid_sizes[i], lw=1, color='b')
        plt.hlines(bid_sizes[i], bid_prices[i + 1], bid_prices[i], lw=1, color='b')

    for i in range(0, len(ask_prices) - 1):
        plt.vlines(ask_prices[i + 1], ask_sizes[i + 1], ask_sizes[i], lw=1, color='r')
        plt.hlines(ask_sizes[i], ask_prices[i + 1], ask_prices[i], lw=1, color='r')

    mid_price = (snapshot[0][0][0] + snapshot[1][0][0]) / 2
    biggest_price = snapshot[1][-1][0]
    lowest_price = snaphsot[0][0][0]
    step = (biggest_price - lowest_price) / 20
    current_value = (biggest_price - lowest_price) / 20

    while step < biggest_price:
        plt.axvspan(current_value, current_value + step, i + 1, facecolor='g', alpha=0.5)
        if i in negative_ind:
            plt.axvspan(i, i + 1, facecolor='r', alpha=0.5)

    bids_legend = mpatches.Patch(color='blue', label='Cumulated bid prices')
    asks_legend = mpatches.Patch(color='red', label='Cumulated ask prices')
    plt.legend(handles=[asks_legend, bids_legend])
    plt.savefig('./doc-figs/cumulated_lob.png')
    # plt.plot(bid_prices, bid_sizes)
    # plt.plot(ask_prices, ask_sizes)
    plt.show()

# plot_mid_prices(np.load('/home/ralph/dev/smartlab/data/snapshots/depth-100/BTC-USD/2020-04-05_0.npy'), 'Price of BTC-USD, 2020. 04. 05. 10 samples used for labelling', 12)
plot_snapshot(np.load('/home/ralph/dev/smartlab/data/snapshots/may-depth-40/BTC-USD/2020-05-06_0.npy'))

