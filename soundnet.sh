#!/bin/bash
source ./config.sh

cd $DATA_DIR

# feature extract 
# and save the results in /raw
mkdir raw/
python3 -u $BASE_DIR/soundnet/extract_feats.py -m $BASE_DIR/soundnet/models/sound8.npy -i ./mp3 -o ./raw -f .mp3

# post processing to get feature vectors
## y_scns
mkdir avg_pooling/ avg_pooling/y_scns/
python3 -u $BASE_DIR/soundnet/get_avg_pool.py -i ./raw -o ./avg_pooling/y_scns -f y_scns

## y_obj
mkdir avg_pooling/y_obj
python3 -u $BASE_DIR/soundnet/get_avg_pool.py -i ./raw -o ./avg_pooling/y_obj -f y_obj

## conv7
mkdir avg_pooling/conv7
python3 -u $BASE_DIR/soundnet/get_avg_pool.py -i ./raw -o ./avg_pooling/conv7 -f conv7