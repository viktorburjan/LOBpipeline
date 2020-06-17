import datetime
import gzip
import os
import time
from cbpro import PublicClient
import json

import schedule

from utils import setup_custom_logger
from utils import create_snapshot_string

# Do we want to get fixed currency-pairs or dynamic?
CONFIG_MODE = 'fixed'

logger = setup_custom_logger('snapshot-crawler')

product_ids = ['BTC-USD', 'ETH-USD', 'BTC-EUR', 'XRP-USD', 'EOS-USD']
client = PublicClient()

targets = {pr_id: gzip.open(create_snapshot_string(pr_id), 'a+') for pr_id in product_ids}


def job(target_csv_files):
    logger.info('Opening new csv files for the next day')
    for csv_file in target_csv_files:
        csv_file.close()
        target_csv_files.clear()
        for prod_id in product_ids:
            targets[prod_id] = gzip.open(create_snapshot_string(prod_id), 'a+')
    return


def save_snapshot():
    logger.info('Saving snapshot for products at %s to files to %s', str(datetime.datetime.now()), str(targets))

    for prod_id in product_ids:
        targets[prod_id].write((json.dumps(client.get_product_order_book(prod_id, 3)) + '\n').encode('utf-8'))


schedule.every(30).seconds.do(save_snapshot)
schedule.every().day.at("00:00").do(job, targets)

logger.info('Starting application main loop')
while True:
    schedule.run_pending()
    time.sleep(1)
