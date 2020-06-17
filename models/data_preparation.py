import datetime
import gzip
import os
import time
from itertools import islice

import ciso8601
import numpy as np
import orjson
from sortedcontainers import SortedList

# %% md

## Utility functions

# %%

# # Configurable fields
# figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

TOTAL_SNAPSHOTS_SAVED = 0
LIMIT = 100000000


def to_float(s):
    return float(s)


def map_snapshots(bids_saved, asks_saved, lob_depth):
    new_snapshot = [bids_saved, asks_saved]
    # Each order also contains an order id, which we don't need. Also converting values to float.
    for i in range(0, lob_depth):
        new_snapshot[0][i] = [float(new_snapshot[0][i][0]), float(new_snapshot[0][i][1])]
        new_snapshot[1][i] = [float(new_snapshot[1][i][0]), float(new_snapshot[1][i][1])]
    return new_snapshot


# Used to extract the parts which we save to the 'processed' folder.
def save_snapshot(sorted_bids, sorted_asks, lob_depth):
    return map_snapshots(sorted_bids[:-1 * lob_depth - 1:-1], sorted_asks[:lob_depth], lob_depth)


class DataPreparation:

    def __init__(self, is_time, nr_of_steps, lob_depth, processing_depth):
        """
        :param is_time: Is the diff between snapshots time or steps?
        :param nr_of_steps: seconds/nr. or orders between 2 saved snapshots
        """
        self.current_order_file_iter = {'day': '', 'iter': None}
        self.is_time = is_time
        self.nr_of_steps = nr_of_steps
        self.processed_updates = 0
        self.lob_depth = lob_depth
        self.processing_depth = processing_depth
        self.TOTAL_SNAPSHOTS_SAVED = 0

    # Applies one update message from the csv file to the order book.
    def apply_to_orderbook(self, sorted_bids, sorted_asks, bids, asks, update):
        if update['type'] == 'open':
            if update['side'] == 'buy':
                new_buy_order = [float(update['price']), float(update['remaining_size']),
                                 update['order_id']]  # price - (remaining_) size - order_id
                sorted_bids.add(new_buy_order)
                bids[update['order_id']] = new_buy_order
            elif update['side'] == 'sell':
                new_sell_order = [float(update['price']), float(update['remaining_size']), update['order_id']]
                sorted_asks.add(new_sell_order)
                asks[update['order_id']] = new_sell_order

        elif update['type'] == 'done':
            bid_removed = bids.pop(update['order_id'], False)
            if bid_removed:
                sorted_bids.remove(bid_removed)
            ask_removed = asks.pop(update['order_id'], False)
            if ask_removed:
                sorted_asks.remove(ask_removed)

            # This block was to introduce checking of consistency in our orderbook. It would not work, because we'd need
            # to store the 'received' orders also (a 'done' or 'match' can reference to an order which is only 'received')

            # if not bid_removed and not ask_removed and update[3]:
            #    if update[3] in received_orders:
            #        received_orders.remove(update[3])
            #    else:
            #        print('Inconsistent orderbook. Removing inconsistent order ', update[3])

        # When we have a match, it may not be in the order book, because it gets filled immediately.
        elif update['type'] == 'match':
            if update['maker_order_id'] in bids:  # maker_order_id
                maker_order = bids[update['maker_order_id']]
                new_size = float(maker_order[1]) - float(update['size'])  # size
                if new_size <= 0:
                    bid_removed = bids.pop(update['maker_order_id'], False)
                    sorted_bids.remove(bid_removed)
                else:
                    maker_order[1] = new_size

            elif update['maker_order_id'] in asks:  # maker_order_id
                maker_order = asks[update['maker_order_id']]
                new_size = float(maker_order[1]) - float(update['size'])  # size
                if new_size <= 0:
                    ask_removed = asks.pop(update['maker_order_id'], False)
                    sorted_asks.remove(ask_removed)
                else:
                    maker_order[1] = new_size

        elif update['type'] == 'change':
            order_id = update['order_id']
            if order_id in bids:
                bids[order_id][1] = float(update['new_size'])  # size = new_size
            elif order_id in asks:
                asks[order_id][1] = update['new_size']  # size = new_size

        elif update['type'] == 'activate':
            print('Not implementing stop orders yet.')

    # For the order book data, we need 2 'views':
    #   - Indexed by id - to be able to apply the updates fast (O(1) lookup)
    #   - Sorted by price - to be able to get the first n orders for snapshots
    # We create maps by ids, and sorted lists by prices here.
    def create_map(self, bids_input, asks_input):
        sorted_bids = SortedList(bids_input, key=lambda x: float(x[0]))
        sorted_asks = SortedList(asks_input, key=lambda x: float(x[0]))

        sorted_bidmap = {}
        sorted_askmap = {}

        for bid in sorted_bids:
            sorted_bidmap[bid[2]] = bid

        for ask in sorted_asks:
            sorted_askmap[ask[2]] = ask

        return sorted_bids, sorted_asks, sorted_bidmap, sorted_askmap

    i = 0

    # train_snapshots: array of snapshots for one day
    @staticmethod
    def write_day_to_disk(out_folder, currency, train_snapshots, date_str, day_index):
        snapshots_to_train = np.array(train_snapshots)

        print('Gonna write data with shapes ', snapshots_to_train.shape)

        if not os.path.exists(os.path.join(out_folder, currency)):
            print('Creating dirs for ', currency)
            os.makedirs(os.path.join(out_folder, currency))

        np.save(os.path.join(out_folder, currency, date_str + "_" + str(day_index)), snapshots_to_train)
        print('Done with part {} of the day {}'.format(day_index, date_str))

    def create_snapshots(self, start_date, end_date, base_dir, output_dir):


        original_start_date = start_date

        updates_base = os.path.join(base_dir, 'updates')
        snapshots_base = os.path.join(base_dir, 'snapshots')

        for currency in os.listdir(updates_base):
            if os.path.exists(os.path.join(output_dir, currency)):
                print('The output folder already exists for currency {}, not running snapshot generation'.format(currency))
                return

            start_date = original_start_date
            print('Parsing ', currency, ' - from ', start_date)

            if currency != 'BTC-USD':
                continue

            while start_date < end_date:
                try:
                    day_index = 0
                    print('Parsing date ', str(start_date), ' for currency ', currency)
                    train_snapshots = []
                    train_seqs = []

                    curr_snapshot_file = os.path.join(snapshots_base, currency,
                                                      start_date.strftime(
                                                          "%Y-%m-%d") + '-' + currency + '__Snapshot.gz')
                    snapshots = islice(iter(gzip.open(curr_snapshot_file, 'rb')), 0, None, 1)

                    curr_update_file = os.path.join(updates_base, currency,
                                                    currency + '__L3Update.log.' + start_date.strftime("%Y-%m-%d") + '.gz')
                    updates = iter(gzip.open(curr_update_file, 'r'))

                    snapshot = orjson.loads(next(snapshots).decode('UTF-8'))
                    next_snapshot = orjson.loads(next(snapshots).decode('UTF-8'))
                    next_seq = next_snapshot['sequence']
                    update = orjson.loads(next(updates)[2:-2])

                    # We should get into a consistent state, where the seq. of the snapshot and the updates are the same,
                    # or else there could be orders stuck until we get the next real snapshot from the raw data.
                    while update['sequence'] > snapshot['sequence']:
                        snapshot = next_snapshot
                        next_snapshot = orjson.loads(next(snapshots).decode('UTF-8'))

                    while True:
                        update = orjson.loads(next(updates)[2:-2])
                        if update['sequence'] >= snapshot['sequence']:
                            break

                    bids = snapshot['bids'][0:self.processing_depth]
                    asks = snapshot['asks'][0:self.processing_depth]
                    sorted_bids, sorted_asks, bidmap, askmap = self.create_map(bids, asks)

                    last_update_time = ciso8601.parse_datetime(update['time'])

                    while True:
                        update = next(updates, None)
                        if update is None:
                            break
                        update = orjson.loads(update[2:-2])
                        self.apply_to_orderbook(sorted_bids, sorted_asks, bidmap, askmap, update)
                        self.processed_updates += 1
                        # Currently not really used
                        if not self.is_time:
                            if self.processed_updates % 2000 == 0:
                                new_snapshot = save_snapshot(sorted_bids, sorted_asks, self.lob_depth)
                                train_snapshots.append(new_snapshot)
                                train_seqs.append(update['sequence'])
                        # Time-based gaps are used instead:
                        else:
                            current_update_time = ciso8601.parse_datetime(update['time'])
                            time_delta = current_update_time - last_update_time
                            # We stop saving more into the current file, if the time gap is bigger than 10 sec.
                            if time_delta > datetime.timedelta(seconds=10):
                                print('Time delta bigger than 10 secs.')
                                self.write_day_to_disk(output_dir, currency, train_snapshots,
                                                       start_date.strftime("%Y-%m-%d"),
                                                       day_index)
                                self.TOTAL_SNAPSHOTS_SAVED += len(train_snapshots)
                                train_snapshots = []
                                day_index += 1
                                maybe_snapshot = next(snapshots, None)
                                if maybe_snapshot is not None:
                                    next_snapshot = orjson.loads(maybe_snapshot.decode('UTF-8'))
                                    next_seq = next_snapshot['sequence']
                                    bids = snapshot['bids'][0:self.processing_depth]
                                    asks = snapshot['asks'][0:self.processing_depth]
                                    sorted_bids, sorted_asks, bidmap, askmap = self.create_map(bids, asks)
                                    last_update_time = current_update_time
                                else:
                                    break
                            elif time_delta > datetime.timedelta(seconds=self.nr_of_steps):
                                if len(sorted_bids) < self.lob_depth or len(sorted_asks) < self.lob_depth:
                                    print('Not enough orders in book')
                                    self.write_day_to_disk(output_dir, currency, train_snapshots,
                                                           start_date.strftime("%Y-%m-%d"),
                                                           day_index)
                                    self.TOTAL_SNAPSHOTS_SAVED += len(train_snapshots)
                                    train_snapshots = []
                                    day_index += 1
                                    maybe_snapshot = next(snapshots, None)
                                    if maybe_snapshot is not None:
                                        next_snapshot = orjson.loads(maybe_snapshot.decode('UTF-8'))
                                        next_seq = next_snapshot['sequence']
                                        bids = snapshot['bids'][0:self.processing_depth]
                                        asks = snapshot['asks'][0:self.processing_depth]
                                        sorted_bids, sorted_asks, bidmap, askmap = self.create_map(bids, asks)
                                        last_update_time = current_update_time

                                        while True:
                                            update = orjson.loads(next(updates)[2:-2])
                                            if update['sequence'] > snapshot['sequence']:
                                                break
                                        continue
                                    else:
                                        break

                                else:
                                    new_snapshot = save_snapshot(sorted_bids, sorted_asks, self.lob_depth)
                                    train_snapshots.append(new_snapshot)
                                    last_update_time = current_update_time

                        if update['sequence'] >= next_seq:
                            snapshot = next_snapshot
                            maybe_snapshot = next(snapshots, None)
                            if maybe_snapshot is not None:
                                next_snapshot = orjson.loads(maybe_snapshot.decode('UTF-8'))
                                next_seq = next_snapshot['sequence']
                                bids = snapshot['bids'][0:self.processing_depth]
                                asks = snapshot['asks'][0:self.processing_depth]
                                sorted_bids, sorted_asks, bidmap, askmap = self.create_map(bids, asks)
                            else:
                                break
                        if self.processed_updates > LIMIT:
                            # a horrible hack so we don't enter the loop again
                            start_date = start_date + datetime.timedelta(weeks=10000)
                            break
                    self.write_day_to_disk(output_dir, currency, train_snapshots, start_date.strftime("%Y-%m-%d"),
                                           day_index)
                    self.TOTAL_SNAPSHOTS_SAVED += len(train_snapshots)
                except FileNotFoundError as ex:
                    print('A file was not found, continuing with next date. Current date was: {}, ex: {}'.format(
                        start_date.strftime("%Y-%m-%d"), ex))
                except (StopIteration, Exception) as ex:
                    print('Unexpected exception has happened: {}'.format(ex))
                    if train_snapshots is not None:
                        self.write_day_to_disk(output_dir, currency, train_snapshots, start_date.strftime("%Y-%m-%d"),
                                               day_index)
                        self.TOTAL_SNAPSHOTS_SAVED += len(train_snapshots)
                        train_snapshots = []
                        day_index += 1
                        print('A gzip file has ended unexpectedly for date: {}'.format(start_date.strftime("%Y-%m-%d")))
                finally:
                    start_date = start_date + datetime.timedelta(days=1)

# %%
#
start = time.time()
lob_depth = 40
data_preparator = DataPreparation(True, 1, lob_depth, lob_depth * 16)

raw_input_folder = '/home/ralph/dev/smartlab/data/compressed_may/'

processed_output_folder = os.path.join('/home/ralph/dev/smartlab/data/snapshots/', 'may-depth-' + str(lob_depth))

data_preparator.create_snapshots(datetime.datetime.strptime('2020-05-05', "%Y-%m-%d"),
                                 datetime.datetime.strptime('2020-05-17', "%Y-%m-%d"), raw_input_folder,
                                 processed_output_folder)
end = time.time()
print('Took to finish: ')
print(end - start)
print('Saved {} snapshots'.format(data_preparator.TOTAL_SNAPSHOTS_SAVED))
print('Total number of updates processeperford: {}'.format(data_preparator.processed_updates))
# %%


# %%
