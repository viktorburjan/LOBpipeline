from numba import jit
import datetime
import gzip
import orjson
import os
import time

import numpy as np
import pandas
from matplotlib.pyplot import figure
from sortedcontainers import SortedList
import ciso8601

from itertools import islice


def create_snapshots(start_date, base_dir):
    updates_base = os.path.join(base_dir, 'updates')
    snapshots_base = os.path.join(base_dir, 'snapshots')

    for currency in os.listdir(base_dir):
        curr_input_file = os.path.join(base_dir, currency,
                                       start_date.strftime(
                                           "%Y-%m-%d") + '-' + currency + '__L3Update.gz')

        curr_date = start_date
        output_file = gzip.open(os.path.join(base_dir, currency,
                                             start_date.strftime(
                                                 "%Y-%m-%d-2") + '-' + currency + '__L3Update.gz'), mode='wb')

        if currency == 'processed' or currency == 'normalised':
            continue

        for line in gzip.open(curr_input_file):
            line_json = orjson.loads(line.decode('UTF-8'))
            time_of_order = ciso8601.parse_datetime(line_json['time'])
            if time_of_order.date() == curr_input_file:
                output_file.write(line)
            else:
                output_file.close()
                start_date = ciso8601.parse_datetime(line_json['time'])
                output_file = gzip.open(os.path.join(base_dir, currency,
                                                     'JSON__' + start_date.strftime(
                                                         "%Y-%m-%d-2") + '-' + currency + '__L3Update.gz'))


create_snapshots(datetime.datetime.strptime('2020-03-28', "%Y-%m-%d"), '/home/ralph/dev/smartlab/data/raw/')
