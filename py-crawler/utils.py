import datetime
import logging
import os
import random
import string
import sys

from cbpro import PublicClient


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def create_csv_string(prod_id, qualifier_and_extension):
    folder = './data-2020/snapshots/' + prod_id + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder + datetime.date.today().strftime('%Y-%m-%d') + '-' + prod_id + '__' + qualifier_and_extension


def create_update_string(prod_id):
    return create_csv_string(prod_id, 'L3Update.gz')


def create_snapshot_string(prod_id):
    return create_csv_string(prod_id, 'Snapshot.gz')


def setup_custom_logger(name):
    log_path = './logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('logs/' + name + '-log.txt', mode='w')
    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    new_logger = logging.getLogger(name)
    new_logger.setLevel(logging.DEBUG)
    new_logger.addHandler(handler)
    new_logger.addHandler(screen_handler)
    for handler in new_logger.handlers:
        handler.setFormatter(formatter)
    return new_logger
