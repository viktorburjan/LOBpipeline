import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy.lib.stride_tricks
from matplotlib.pyplot import *
from numba import cuda
from numba import jit


@jit(nopython=True)
def new_digitize(x, bins):
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bins must be monotonically increasing or decreasing")

    # this is backwards because the arguments below are swapped

    return np.searchsorted(bins, x, side='right')


# https://stackoverflow.com/questions/53263678/generalized-method-for-rolling-or-sliding-window-over-array-axis
def as_sliding_window(x, window_size, axis=0, window_axis=None, ):
    """
    Make a sliding window across an axis.

    Uses ``numpy.lib.stride_tricks.as_strided``, similar caveats apply.

    Parameters
    ----------
    x : array_like
        Array from where the sliding window is created.
    window_size: int
        Size of the sliding window.
    axis: int
        Dimension across which the sliding window is created.
    window_axis: int
        New dimension for the sliding window. By default, the new
        dimension is inserted before ``axis``.
    subok: bool
        If True, subclasses are preserved
        (see ``numpy.lib.stride_tricks.as_strided``).
    writeable: bool
        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible
        (see ``numpy.lib.stride_tricks.as_strided``).

    Returns
    --------
    sliding_window: ndarray
        View of the given array as a sliding window along ``axis``.
    """
    x = np.asarray(x)
    axis %= x.ndim
    if window_axis is None:
        window_axis = axis
    window_axis %= x.ndim + 1
    # Make shape
    shape = list(x.shape)
    n = shape[axis]
    shape[axis] = window_size
    shape.insert(window_axis, max(n - window_size + 1, 0))
    # Make strides
    strides = list(x.strides)
    strides.insert(window_axis, strides[axis])
    # Make sliding window view
    sliding_window = numpy.lib.stride_tricks.as_strided(x, shape, strides)
    return sliding_window


fig = plt.figure()
start_time = time.time()

timings_divide = []
timings_binning = []


@cuda.jit
def my_search_sorted(prices, bins):
    indexes = np.empty(len(prices), dtype=np.int64)
    for i in range(0, len(prices)):
        # real item:
        current_item = prices[i]
        if current_item < bins[0]:
            indexes[i] = 0
        for j in range(0, len(bins)):
            if bins[j] <= current_item < bins[j + 1]:
                indexes[i] = j
    return indexes


@cuda.jit
def my_cumsum(values):
    new_values = np.zeros(len(values), dtype=np.int64)
    new_values[0] = values[0]
    for i in range(1, len(values)):
        new_values[i] = new_values[i - 1] + values[i]
    return new_values


# digitized bids: (nr_of_batches, batch_size, bin_size)

def bin_batch(nr_of_bins, batch_size, done_bids, done_asks, new_array):
    # done_bids.shape[0] is the number of batches that we are processing
    # new_array = np.zeros((done_bids.shape[0], batch_size, 2, nr_of_bins))  # TODO
    for batch_index in range(0, done_bids.shape[0]):
        for snapshot_index in range(0, batch_size):
            new_array[batch_index][snapshot_index] += np.cumsum(done_bids[batch_index][snapshot_index][::-1])[::-1]
            new_array[batch_index][snapshot_index] += np.cumsum(done_asks[batch_index][snapshot_index])


@jit(nopython=True)
def average(x):
    return (x[0][0][0] + x[1][0][0]) / 2


# TODO Write this with CUDA
@jit(nopython=True)
def calculate_labels_mid_prices(current_snapshots, labels_lookahead):
    mid_prices = []
    for i in range(0, len(current_snapshots)):
        mid_prices.append(average(current_snapshots[i]))
    mid_prices = np.array(mid_prices)
    # labels = []
    #
    # for i in range(0, len(current_snapshots) - labels_lookahead):
    #     next_midprices = mid_prices[i + 1: i + labels_lookahead]
    #     next_midprices_avg = np.sum(next_midprices) / len(next_midprices)
    #     labels.append(mid_prices[i] < next_midprices_avg)
    return mid_prices


def calculate_vwap(current_snapshots):
    vwaps = []
    for i in range(0, len(current_snapshots)):
        price_dot_size_bid = current_snapshots[i, 0, :, 0] * current_snapshots[i, 0, :, 1]
        price_dot_size_ask = current_snapshots[i, 1, :, 0] * current_snapshots[i, 1, :, 1]
        price_volume_sum = sum(price_dot_size_bid + price_dot_size_ask)
        sum_sizes = np.sum(current_snapshots[i, 0, :, 1]) + np.sum(current_snapshots[i, 1, :, 1])
        vwaps.append(price_volume_sum / sum_sizes)
    return np.array(vwaps)


@jit(nopython=True)
def get_digitized_batches(batches, bins, digitized_bids, digitized_asks, done_bids, done_asks):
    for batch_index in range(0, len(batches)):
        batch_to_scale = batches[batch_index]
        for snapshot_index in range(0, len(batch_to_scale)):
            bid_prices = batch_to_scale[snapshot_index, 0, :, 0]
            bid_sizes = batch_to_scale[snapshot_index, 0, :, 1]

            ask_prices = batch_to_scale[snapshot_index, 1, :, 0]
            ask_sizes = batch_to_scale[snapshot_index, 1, :, 1]

            for i in range(0, len(bid_prices)):
                current_item = bid_prices[i]
                if current_item < bins[0]:
                    digitized_bids[batch_index][snapshot_index][i] = 0
                    break
                if current_item > bins[-1]:
                    digitized_bids[batch_index][snapshot_index][i] = len(bins) - 2
                    break
                for j in range(0, len(bins)):
                    if bins[j] <= current_item < bins[j + 1]:
                        digitized_bids[batch_index][snapshot_index][i] = j
                        break

            for i in range(0, len(ask_prices)):
                current_item = ask_prices[i]
                if current_item < bins[0]:
                    digitized_asks[batch_index][snapshot_index][i] = 0
                    break
                if current_item > bins[-1]:
                    digitized_asks[batch_index][snapshot_index][i] = len(bins) - 2
                    break
                for j in range(0, len(bins)):
                    if bins[j] <= current_item < bins[j + 1]:
                        digitized_asks[batch_index][snapshot_index][i] = j
                        break

            for nr in range(0, len(bid_sizes)):
                curr_bid_bin = digitized_bids[batch_index][snapshot_index][nr]
                curr_ask_bin = digitized_asks[batch_index][snapshot_index][nr]

                done_bids[batch_index][snapshot_index][curr_bid_bin] += bid_sizes[nr]
                done_asks[batch_index][snapshot_index][curr_ask_bin] += ask_sizes[nr]


@jit(nopython=True)
def normalize_batches(batches, mid_prices_for_batch, lob_depth, target_array):
    """
    :param mid_prices_for_batch: Always the mid price for the last element of the current batch, which we normalise with.
    """
    for batch_nr in range(0, len(batches)):
        batch = batches[batch_nr]
        actual_last_midprice = mid_prices_for_batch[batch_nr]
        for i in range(0, lob_depth):
            for inside_batch_nr in range(0, len(target_array[batch_nr])):
                target_array[batch_nr][inside_batch_nr][0][i][0] = batch[inside_batch_nr][0][i][
                                                                       0] / actual_last_midprice
                target_array[batch_nr][inside_batch_nr][0][i][1] = batch[inside_batch_nr][0][i][1]
                target_array[batch_nr][inside_batch_nr][1][i][0] = batch[inside_batch_nr][1][i][
                                                                       0] / actual_last_midprice
                target_array[batch_nr][inside_batch_nr][1][i][1] = batch[inside_batch_nr][1][i][1]


class LOBNormaliser:
    bins = None

    def __init__(self, lob_depth, is_log, bin_start, bin_end, log_base, nr_of_bins, batch_size, labels_lookahead,
                 bin_step):
        self.lob_depth = lob_depth
        self.is_log = is_log
        self.bin_start = bin_start
        self.bin_end = bin_end
        self.log_base = log_base
        self.nr_of_bins = nr_of_bins
        self.batch_size = batch_size
        self.labels_lookahead = labels_lookahead
        self.bin_step = bin_step  # Only used for output file format

        if is_log:
            self.get_log_bins()
        else:
            self.get_linear_bins()

    def get_log_bins(self):
        if self.bins is None:
            exponential_space = np.logspace(self.bin_start, self.bin_end, base=1.2, num=self.nr_of_bins // 2) - 0.01031
            # exponential_space = np.logspace(-15, -3, base=2, num=NR_OF_BINS // 2)
            first_part_bins = np.flip(np.ones((self.nr_of_bins // 2)) - exponential_space, axis=0)
            second_part_bins = np.array(np.ones((self.nr_of_bins // 2)) + exponential_space)
            self.bins = np.concatenate([first_part_bins, [1], second_part_bins])
            plt.plot(self.bins)
            plt.show()
        return self.bins

    def get_linear_bins(self):
        if self.bins is None:
            self.bins = np.concatenate(
                [np.linspace(self.bin_start, 1, self.nr_of_bins // 2),
                 np.linspace(1, self.bin_end, self.nr_of_bins // 2)])
        return self.bins

    """
        The Orderbook data looks like the following: 
        | bids                              | asks                                     |
        [ [[8454, 0.5], [8453, 0.2], [...]] , [[8450,9], [8454, 1], [...] ]], 
          [[8454, 0.5], [8453, 0.2], [...]] , [[8450,9], [8454, 1], [...] ]],... ]
        This means to find the smallest and largest price, we need to check the 1st elem of each 1-dimensional array inside. 

    """

    @staticmethod
    def calculate_distribution(self, np_snapshots):
        smallest = np.amin(np.reshape(np_snapshots, (len(np_snapshots) * 2 * self.lob_depth, 2)), axis=0)
        largest = np.amax(np.reshape(np_snapshots, (len(np_snapshots) * 2 * self.lob_depth, 2)), axis=0)
        return smallest[0], largest[0]  # 1 would be the smallest/largest size

    def convert_to_np(self, in_snapshots):
        """

        :param in_snapshots: snapshots that we want to convert to numpy array.
               this is the raw input from file read, the numbers are represented as strings.
        :return: np.array of the same data
        """
        # Getting only the first n bids and asks and converting them to ints
        np_snapshots = []
        for snapshot in in_snapshots:
            new_snapshot = [snapshot['bids'][0:self.lob_depth], snapshot['asks'][0:self.lob_depth]]

            # Each order also contains an order, which we don't need. Also converting values to float.
            for i in range(0, self.lob_depth):
                new_snapshot[0][i] = [float(new_snapshot[0][i][0]), float(new_snapshot[0][i][1])]
                new_snapshot[1][i] = [float(new_snapshot[1][i][0]), float(new_snapshot[1][i][1])]
            np_snapshots.append(new_snapshot)

        return np.array(np_snapshots)

    """
    def display_batch(batch):
        """"""
        :param batch: list of snapshots that we would like to display with Matplotlib.
        :return: void, displays the snapshots and saves the picture into a file.
        """"""

        # Reshaping into a single list. We still have price and size,
        # so taking each 2nd element of the list to only have prices for now.
        prices = batch.reshape((LOB_DEPTH * BATCH_SIZE * 4,))[::2]

        # Shaping back the prices into LOB snapshots. Each sub-array is a snapshot.
        to_display = np.sort(prices.reshape((BATCH_SIZE, LOB_DEPTH * 2)), axis=1)

        figure(figsize=(20, 20))
        fig, ax = plt.subplots()
        im = ax.imshow(to_display)
        tight_layout()

        plt.axvline(x=30, lw=4, color='k')

        for vertical_ind in range(BATCH_SIZE):
            for horizontal_ind in range(60):
                ax.text(horizontal_ind, vertical_ind, to_display[vertical_ind][horizontal_ind], ha="center", va="center", color="w", size=5)

        show()
        fig.savefig('order_book.png', dpi=800)
    """

    def open_snapshot_file(self, data_path, snapshot_file_name):
        file_open_start = time.time()
        with gzip.open(data_path + snapshot_file_name) as current_snapshots:
            current_snapshots = list(map(lambda x: json.loads(x.decode('utf-8')), current_snapshots))
            file_open_time = time.time()
            print('Done with opening snapshot file: ', str(snapshot_file_name), ' took ',
                  str(file_open_time - file_open_start))
            print('Depth of bids: ', str(len(current_snapshots[0]['bids'])))

            return self.convert_to_np(current_snapshots)

    def run(self, data_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        ALL_BATCHES_PROCESSED = 0

        # Going through the snapshot files in the directory.
        # TODO this should be something like we have separate folders for each list of snapshots which belong together
        for currency in os.listdir(data_path):
            current_curr_path = os.path.join(data_path, currency)
            for filename in os.listdir(current_curr_path):
                if 'depth' in filename or 'directory' in filename:
                    continue
                print('Opening snapshot file: ', str(filename))

                all_snapshots_loaded = np.load(os.path.join(current_curr_path, filename))
                if len(all_snapshots_loaded) < self.labels_lookahead:
                    print('Not enough samples for day, skipping ', filename)
                    continue
                all_current_snapshots = np.array_split(all_snapshots_loaded, 27)
                print(
                    'Nr. of total snapshots in file : ' + str(len(all_snapshots_loaded)) + ' - dividing into 14 chunks')

                for index_of_chunk, chunk_snapshots in enumerate(all_current_snapshots):
                    current_snapshots = chunk_snapshots[::4]
                    if not current_snapshots.shape[0] > 20:
                        print('The chunk has less than 20 snapshots, skipping')
                        break
                    mid_prices = calculate_labels_mid_prices(current_snapshots, self.labels_lookahead)
                    vwaps = calculate_vwap(current_snapshots)
                    batches = np.ascontiguousarray(as_sliding_window(current_snapshots, self.batch_size))

                    batch_mid_prices = mid_prices[self.batch_size - 1:]

                    threadsperblock = (32, 32)

                    # Calculate the number of thread blocks in the grid
                    blockspergrid_x = int(len(batches) / threadsperblock[0]) + 2
                    blockspergrid_y = int(len(batches[0]) / threadsperblock[1]) + 2

                    nr_of_batches = len(batches)

                    c_target_batches = np.zeros((nr_of_batches, self.batch_size, 2, self.lob_depth, 2))
                    norm_starttime = time.time()

                    print('batches shape: ', str(batches.shape))
                    # c_batches = cuda.to_device(batches)
                    # c_batch_mid_prices = cuda.to_device(batch_mid_prices)

                    normalize_batches(batches, batch_mid_prices,
                                      self.lob_depth,
                                      c_target_batches)
                    target_batches = c_target_batches

                    timings_divide.append(time.time() - norm_starttime)
                    # batches are normalised with the last mid-price

                    digitized_bids = np.zeros((nr_of_batches, self.batch_size, self.lob_depth), dtype=np.int64)
                    digitized_asks = np.zeros((nr_of_batches, self.batch_size, self.lob_depth), dtype=np.int64)

                    done_bids = np.zeros((nr_of_batches, self.batch_size, self.nr_of_bins))
                    done_asks = np.zeros((nr_of_batches, self.batch_size, self.nr_of_bins))

                    get_digitized_batches(target_batches, self.bins, digitized_bids, digitized_asks, done_bids,
                                          done_asks)

                    # done_bids = done_bids.copy_to_host()
                    # done_asks = done_asks.copy_to_host()

                    # snapshots_to_display = np.zeros((len(target_batches), self.batch_size, 2, self.nr_of_bins))
                    # bin_batch(target_batches, self.bins, done_bids, done_asks, new_array, snapshots_to_display)

                    new_array = np.zeros((done_bids.shape[0], self.batch_size, self.nr_of_bins))
                    bin_batch(self.nr_of_bins, self.batch_size, done_bids, done_asks, new_array)

                    out_currency_folder = os.path.join(output_path, 'linear-' + str(self.bin_step) + '-' + str(
                        self.nr_of_bins) + '_bins', currency)
                    if not os.path.exists(out_currency_folder):
                        os.makedirs(out_currency_folder)

                    batch_out_file = os.path.join(out_currency_folder, filename.split('.')[0] + '-normalised-' + str(
                        index_of_chunk) + '.npy')
                    mid_price_out_file = os.path.join(out_currency_folder, filename.split('.')[0] + '-mid_price-' + str(
                        index_of_chunk) + '.npy')
                    vwap_out_file = os.path.join(out_currency_folder,
                                                 filename.split('.')[0] + '-vwap-' + str(index_of_chunk) + '.npy')

                    # for i in range(0,200):
                    #     plt.clf()
                    #     plt.title(str(self.nr_of_bins) + ' bins, log scale, ' + str(self.lob_depth) + ' LOB depth. ' + str(10))
                    #     plt.imshow(new_array[i], aspect='auto', cmap=plt.get_cmap('hot'),
                    #                extent=[self.bins[0], self.bins[-1], 0, self.batch_size])
                    #     # plt.clim(0, 50)
                    #     plt.xlabel('Price deviance to mid price (ratio)')
                    #     plt.ylabel('Snapshots index in batch')
                    #     plt.xticks([self.bins[0], self.bins[-1]], visible=True, rotation="horizontal")
                    #     plt.yticks(np.arange(1, self.batch_size), visible=True, rotation=45)
                    #     plt.colorbar()
                    #     # plt.show()
                    #     plt.savefig('./figs/snapshot-' + str(i) + '.png')
                    print('Saving normalised batch to ', str(batch_out_file))

                    np.save(batch_out_file, np.array(new_array))
                    np.save(mid_price_out_file, batch_mid_prices)
                    np.save(vwap_out_file, np.array(vwaps[self.batch_size - 1:]))

                    ALL_BATCHES_PROCESSED += len(new_array)

        end_time = time.time()
        print('Processing took {} for {} batches. LOB depth: {}, batch size: {}'.format(str(end_time - start_time),
                                                                                        str(ALL_BATCHES_PROCESSED),
                                                                                        self.lob_depth,
                                                                                        self.batch_size))
        print('times of divide func without optimization: ', timings_divide)


BATCH_SIZE = 25
LOB_DEPTH = 40
LABELS_LOOKAHEAD = 5

# BIN_START = 0.997
BIN_START = -25
# BIN_END = 1.003
BIN_END = -23
BIN_STEP = 0.0002

NR_OF_BINS = 50

normaliser = LOBNormaliser(LOB_DEPTH, True, BIN_START, BIN_END, 1.2, NR_OF_BINS, BATCH_SIZE, LABELS_LOOKAHEAD, BIN_STEP)

normaliser.run('/home/ralph/dev/smartlab/data/snapshots/depth-' + str(LOB_DEPTH),
               '/home/ralph/dev/smartlab/data/normalised_JSON/depth-' + str(LOB_DEPTH))

# fig, ax = plt.subplots()
# if self.is_log:
#     for sn_ind in range(0, 600):
#         plt.clf()
#         plt.title(str(self.nr_of_bins) + ' bins, log scale, ' + str(self.lob_depth) + ' LOB depth. ' + str(sn_ind))
#         plt.imshow(new_array[sn_ind], aspect='auto', cmap=plt.get_cmap('hot'),
#                    extent=[self.bins[0], self.bins[-1], 0, 10])
#         # plt.clim(0, 50)
#         plt.xlabel('Price deviance to mid price (ratio)')
#         plt.ylabel('Snapshots index in batch')
#         plt.xticks([self.bins[0], self.bins[-1]], visible=True, rotation="horizontal")
#         plt.yticks(np.arange(1, 11), visible=True, rotation=45)
#         plt.colorbar()
#         plt.savefig('./figs/' + currency + '_' + str(filename[0:10]) + '-' + str(sn_ind) + '-bins-' + str(
#             self.nr_of_bins) + 'log.png')
#         # plt.show()
# else:
#     plt.title(str(self.nr_of_bins) + ' bins, lin scale, ' + str(self.lob_depth) + ' LOB depth.')
#     plt.imshow(new_array[290], aspect='auto', cmap=plt.get_cmap('hot'), extent=[self.bins[0], self.bins[-1], 0, 10])
#     plt.clim(0, 50)
#     plt.xlabel('Price deviance to mid price (ratio)')
#     plt.ylabel('Snapshots index in batch')
#     plt.xticks([self.bins[0], self.bins[-1]], visible=True, rotation="horizontal")
#     plt.yticks(np.arange(1, 11), visible=True, rotation=45)
#     plt.colorbar()
#     plt.savefig(
#         './figs/' + str(filename[0:10]) + '-' + str(index_of_chunk) + '-bins-' + str(self.nr_of_bins) + 'lin.png')
#     plt.show()
