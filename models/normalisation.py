import gzip
import json
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numba import cuda
from numba import jit


@jit(nopython=True)
def new_digitize(x, bins):
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bins must be monotonically increasing or decreasing")
    return np.searchsorted(bins, x, side='right')

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


@jit(nopython=True)
def my_cumsum(values):
    new_values = np.zeros(len(values), dtype=np.int64)
    new_values[0] = values[0]
    for i in range(1, len(values)):
        new_values[i] = new_values[i - 1] + values[i]
    return new_values


# TODO write this into CUDA, which is not that easy
@jit(nopython=True)
def bin_batch(batch_size, done_bids, done_asks, new_array):
    # done_bids.shape[0] is the number of batches that we are processing
    # new_array = np.zeros((done_bids.shape[0], batch_size, 2, nr_of_bins))  # TODO
    for batch_index in range(0, done_bids.shape[0]):
        for snapshot_index in range(0, batch_size):
            new_array[batch_index][snapshot_index] += np.cumsum(done_bids[batch_index][snapshot_index][::-1])[::-1]
            new_array[batch_index][snapshot_index] += np.cumsum(done_asks[batch_index][snapshot_index])


@jit(nopython=True)
def average(x):
    return (x[0][0][0] + x[1][0][0]) / 2


def calculate_vwap(current_snapshots):
    vwaps = []
    for i in range(0, len(current_snapshots)):
        price_dot_size_bid = current_snapshots[i, 0, :, 0] * current_snapshots[i, 0, :, 1]
        price_dot_size_ask = current_snapshots[i, 1, :, 0] * current_snapshots[i, 1, :, 1]
        price_volume_sum = sum(price_dot_size_bid + price_dot_size_ask)
        sum_sizes = np.sum(current_snapshots[i, 0, :, 1]) + np.sum(current_snapshots[i, 1, :, 1])
        vwaps.append(price_volume_sum / sum_sizes)
    return np.array(vwaps)


@cuda.jit()
def get_digitized_batches(batches, bins, digitized_bids, digitized_asks, done_bids, done_asks):
    batch_index, snapshot_index = cuda.grid(2)
    if batch_index < len(batches):
        batch_to_scale = batches[batch_index]
        if snapshot_index < len(batch_to_scale):
            bid_prices = batch_to_scale[snapshot_index, 0, :, 0]
            bid_sizes = batch_to_scale[snapshot_index, 0, :, 1]

            ask_prices = batch_to_scale[snapshot_index, 1, :, 0]
            ask_sizes = batch_to_scale[snapshot_index, 1, :, 1]

            for i in range(0, len(bid_prices)):
                current_item = bid_prices[i]
                if current_item < bins[0]:
                    digitized_bids[batch_index][snapshot_index][i] = 0
                    break
                if current_item > bins[len(bins) - 1]:
                    digitized_bids[batch_index][snapshot_index][i] = len(bins) - 2
                    break
                for j in range(0, len(bins) - 2):
                    if bins[j] <= current_item < bins[j + 1]:
                        digitized_bids[batch_index][snapshot_index][i] = j
                        break

            for i in range(0, len(ask_prices)):
                current_item = ask_prices[i]
                if current_item < bins[0]:
                    digitized_asks[batch_index][snapshot_index][i] = 0
                    break
                if current_item > bins[len(bins) - 1]:
                    digitized_asks[batch_index][snapshot_index][i] = len(bins) - 2
                    break
                for j in range(0, len(bins) - 2):
                    if bins[j] <= current_item < bins[j + 1]:
                        digitized_asks[batch_index][snapshot_index][i] = j
                        break

            for nr in range(0, len(bid_sizes)):
                curr_bid_bin = digitized_bids[batch_index][snapshot_index][nr]
                curr_ask_bin = digitized_asks[batch_index][snapshot_index][nr]

                current_batch = done_bids[batch_index]
                current_snapshot = current_batch[snapshot_index]
                current_snapshot[curr_bid_bin] += bid_sizes[nr]

                done_asks[batch_index][snapshot_index][curr_ask_bin] += ask_sizes[nr]


@cuda.jit
def normalize_batches(snapshots, batch_mid_prices, lob_depth, target_array, batch_size):
    """
    :param mid_prices_for_batch: Always the mid price for the last element of the current batch, which we normalise with.
    """
    batch_nr, inside_batch_nr = cuda.grid(2)
    batch = snapshots[batch_nr:batch_nr + batch_size]

    for i in range(0, lob_depth):
        if batch_nr < len(snapshots) - batch_size and inside_batch_nr < len(target_array[batch_nr]):
            batch_mid_prices[batch_nr] = (snapshots[batch_nr + batch_size - 1][0][0][0] + snapshots[batch_nr + batch_size - 1][1][0][0]) / 2
            target_array[batch_nr][inside_batch_nr][0][i][0] = batch[inside_batch_nr][0][i][0] / batch_mid_prices[batch_nr]
            target_array[batch_nr][inside_batch_nr][0][i][1] = batch[inside_batch_nr][0][i][1]
            target_array[batch_nr][inside_batch_nr][1][i][0] = batch[inside_batch_nr][1][i][0] / batch_mid_prices[batch_nr]
            target_array[batch_nr][inside_batch_nr][1][i][1] = batch[inside_batch_nr][1][i][1]


class LOBNormaliser:
    bins = None

    def __init__(self, lob_depth, is_log, bin_start, bin_end, log_base, nr_of_bins, batch_size, bin_step):
        self.lob_depth = lob_depth
        self.nr_of_bins = nr_of_bins + 1
        self.batch_size = batch_size
        self.bin_step = bin_step  # Only used for output file format

        if is_log:
            self.get_log_bins(log_base, bin_start, bin_end)
        else:
            self.get_linear_bins(bin_start, bin_end)

    def get_log_bins(self, log_base, bin_start, bin_end):
        if self.bins is None:
            exponential_space = np.logspace(bin_start, bin_end, base=log_base, num=self.nr_of_bins // 2) - 0.01031
            # exponential_space = np.logspace(-15, -3, base=2, num=NR_OF_BINS // 2)
            first_part_bins = np.flip(np.ones((self.nr_of_bins // 2)) - exponential_space, axis=0)
            second_part_bins = np.array(np.ones((self.nr_of_bins // 2)) + exponential_space)
            self.bins = np.concatenate([first_part_bins, [1.0], second_part_bins])
            # plt.plot(self.bins)
            # plt.show()
        return self.bins

    def get_linear_bins(self, bin_start, bin_end):
        if self.bins is None:
            self.bins = np.concatenate(
                [np.linspace(bin_start, 1, self.nr_of_bins // 2),
                 np.linspace(1, bin_end, self.nr_of_bins // 2)])
        return self.bins

    def generate_images(self, batch, filename, title):
        plt.clf()
        plt.title(str(self.nr_of_bins) + ' bins, log scale, ' + str(self.lob_depth) + ' ' + title)
        plt.imshow(batch, aspect='auto', cmap=plt.get_cmap('hot'),
                   extent=[self.bins[0], self.bins[-1], 0, self.batch_size])
        # plt.clim(0, 50)
        plt.xlabel('Price deviance to mid price (ratio)')
        plt.ylabel('Snapshots index in batch')
        plt.xticks([self.bins[0], self.bins[-1]], visible=True, rotation="horizontal")
        plt.yticks(np.arange(1, self.batch_size), visible=True, rotation=45)
        plt.colorbar()
        plt.show()
        plt.savefig('./figs/' + filename + '.png')

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
        batches_processed_nr = 0

        for currency in os.listdir(data_path):
            current_curr_path = os.path.join(data_path, currency)
            for filename in os.listdir(current_curr_path):
                if 'depth' in filename or 'directory' in filename:
                    continue
                print('Opening snapshot file: ', str(filename))

                all_snapshots_loaded = np.load(os.path.join(current_curr_path, filename))
                if len(all_snapshots_loaded) < self.batch_size:
                    print('Not enough samples for day, skipping ', filename)
                    continue
                all_current_snapshots = np.array_split(all_snapshots_loaded, 35)
                print('Nr. of total snapshots in file : ' + str(len(all_snapshots_loaded)) + ' - dividing into 17 chunks')

                # This split is only added so that I can run this on my local GPU
                for index_of_chunk, chunk_snapshots in enumerate(all_current_snapshots):
                    current_snapshots = chunk_snapshots
                    if not current_snapshots.shape[0] > 100:
                        print('The chunk has less than 20 snapshots, skipping')
                        break
                    vwaps = calculate_vwap(current_snapshots)

                    threadsperblock = (32, 32)
                    blockspergrid_x = int(len(current_snapshots) / threadsperblock[0]) + 2
                    blockspergrid_y = int(len(current_snapshots[0]) / threadsperblock[1]) + 2

                    # Calculate the number of thread blocks in the grid
                    c_target_batches = cuda.device_array((len(current_snapshots) - self.batch_size, self.batch_size, 2, self.lob_depth, 2))
                    nr_of_batches = len(c_target_batches)
                    c_snapshots = cuda.to_device(np.ascontiguousarray(current_snapshots))

                    norm_starttime = time.time()

                    c_batch_mid_prices = cuda.device_array((nr_of_batches))

                    normalize_batches[(blockspergrid_x, blockspergrid_y), threadsperblock](c_snapshots,
                                                                                           c_batch_mid_prices,
                                                                                           self.lob_depth,
                                                                                           c_target_batches,
                                                                                           self.batch_size)
                    timings_divide.append(time.time() - norm_starttime)
                    batch_mid_prices = c_batch_mid_prices.copy_to_host()

                    # batches are normalised with the last mid-price

                    digitized_bids = cuda.device_array((nr_of_batches, self.batch_size, self.lob_depth), dtype=np.int64)
                    digitized_asks = cuda.device_array((nr_of_batches, self.batch_size, self.lob_depth), dtype=np.int64)

                    done_bids = cuda.device_array((nr_of_batches, self.batch_size, self.nr_of_bins))
                    done_asks = cuda.device_array((nr_of_batches, self.batch_size, self.nr_of_bins))

                    get_digitized_batches[(blockspergrid_x, blockspergrid_y), threadsperblock](c_target_batches,
                                                                                               self.bins,
                                                                                               digitized_bids,
                                                                                               digitized_asks,
                                                                                               done_bids, done_asks)

                    done_bids = done_bids.copy_to_host()
                    done_asks = done_asks.copy_to_host()

                    new_array = np.zeros((done_bids.shape[0], self.batch_size, self.nr_of_bins))
                    bin_batch(self.batch_size, done_bids, done_asks, new_array)

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

                    if self.generate_images:
                        self.generate_images(new_array[10], 'test_image', 'test image')

                    print('Saving normalised batch to ', str(batch_out_file))

                    np.save(batch_out_file, np.array(new_array))
                    np.save(mid_price_out_file, batch_mid_prices)
                    np.save(vwap_out_file, np.array(vwaps[self.batch_size - 1:]))

                    batches_processed_nr += len(new_array)

        end_time = time.time()
        print('Processing took {} for {} batches. LOB depth: {}, batch size: {}'.format(str(end_time - start_time),
                                                                                        str(batches_processed_nr),
                                                                                        self.lob_depth,
                                                                                        self.batch_size))
        print('times of divide func without optimization: ', timings_divide)

#
# BATCH_SIZE = 25
# LOB_DEPTH = 40
#
# # BIN_START = 0.997
# BIN_START = -25
# # BIN_END = 1.003
# BIN_END = -23
# BIN_STEP = 0.0002
#
# NR_OF_BINS = 50
#
# normaliser = LOBNormaliser(LOB_DEPTH, True, BIN_START, BIN_END, 1.2, NR_OF_BINS, BATCH_SIZE, BIN_STEP)
#
# normaliser.run('/home/ralph/dev/smartlab/data/snapshots/may-depth-' + str(LOB_DEPTH),
#                '/home/ralph/dev/smartlab/data/normalised_JSON/depth-' + str(LOB_DEPTH))

# fig, ax = plt.subplots()
# if self.is_log:
#     for sn_ind in range(0, 600):
#         plt.clf()
#         plt.title(str(self.nr_of_bins) + ' bins, log scale, ' + str(self.lob_depth) + ' LOB depth. ' + str(sn_ind))
#         plt.imshow(new_array[sn_ind], aspect='auto', cmap=plt.get_cmap('hot'),
#                    extent=[s[0], self.bins[-1], 0, 10])
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