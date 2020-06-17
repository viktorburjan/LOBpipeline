import datetime
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
import time
import cbpro
import orjson

from utils import setup_custom_logger

product_id_arg = sys.argv[1]

logger = setup_custom_logger('update-crawler_' + product_id_arg)

logger.info('Starting process: ' + product_id_arg)


def create_product_logger(product_id):
    log_path = './data-2020/updates/' + product_id
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    product_handler = TimedRotatingFileHandler(os.path.join(log_path, str(product_id) + '__L3Update.log'),
                                               when='midnight')
    new_logger = logging.getLogger(product_id)
    new_logger.addHandler(product_handler)
    product_handler.setLevel(1)
    new_logger.setLevel(1)
    return new_logger


product_logger = create_product_logger(product_id_arg)


class L3WebsocketClient(cbpro.WebsocketClient):
    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.products = [product_id_arg]
        self.channels = ['heartbeat', 'full']

    def on_message(self, msg):
        try:
            if 'product_id' in msg and msg['type'] != 'heartbeat':
                product_logger.info(orjson.dumps(msg))
        except Exception as ex:
            logger.error(str(ex))

    def on_close(self):
        logger.info("Websocket stream closed.")
        sys.exit(1)

    def on_error(self, e, data=None):
        logger.error('An error has happened!')
        logger.error(e)
        logger.error(data)
        logger.error('----')


wsClient = L3WebsocketClient()
wsClient.start()
time.sleep(10)
sys.exit(0)