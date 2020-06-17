# Coinbase Pro data saver

The 2 scripts - snapshot-crawler and update-crawler - gather data of the Coinbase Pro exchange into the data
folder if launched.

The scripts need the following environmental variables: 

- CONFIG_MODE: fixed or dynamic - how to get the products we want to save? If the mode is dynamic, all the
    products will be saved that exist on the exchange. With 'fixed', the products also need to be supplied.
- PRODUCTS - if the CONFIG_MODE is fixed, the products should be listed here in a coma-separated list
             e.g ETH-USD,BTC-USD

## Running in docker

To run the scripts in a Docker container: 
- Navigate to the folder where this file is
- `sudo docker build -t crawler .`
- `sudo docker run -v {save_data_here}:/data-2020 -p 2223:22 crawler`
  e.g. on deep3: `sudo docker run -v /mnt/hdd1/oracle/crypto-ai/database/event-based/gdax/data-2020:/data-2020 -v /mnt/hdd1/oracle/crypto-ai/database/event-based/gdax/data:/data-old -p 2223:22 crawler`

To SSH into the container: 
`ssh localhost -o  UserKnownHostsFile=/dev/null -p 2223 -l notroot`
