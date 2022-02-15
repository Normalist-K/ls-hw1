#!/bin/bash
DATA_DIR=/shared/youngkim/dataset

python -u get_avg_pool.py -i $DATA_DIR/raw -o $DATA_DIR/avg_pooling/y_scns -f y_scns
python -u get_avg_pool.py -i $DATA_DIR/raw -o $DATA_DIR/avg_pooling/y_obj -f y_obj