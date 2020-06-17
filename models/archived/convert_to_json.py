import os
import gzip
from itertools import islice

files = os.listdir('/home/ralph/dev/smartlab/data/raw/BTC-USD/')
files.sort()
for filename in files:
    gz_file = islice(iter(gzip.open(os.path.join('/home/ralph/dev/smartlab/data/raw/BTC-USD/', filename), encoding='UTF-8', mode='rt')), 0, None, 1)
    out_file = gzip.open(os.path.join('/home/ralph/dev/smartlab/data/JSON/BTC-USD/', "JSON__" + filename), 'wt')
    try:
        while True:
            line = next(gz_file)

            if line is None:
                break
            line = line.replace("'", "\"")
            out_file.write(line)
        out_file.close()
    except Exception as ex:
        print('Exception has happened for{}, {} '.format(filename, str(ex.args)))
        out_file.close()
