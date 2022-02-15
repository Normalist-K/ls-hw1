#!/bin/bash
source ./config.sh
TRAIN_DATA=$DATA_DIR/labels/train_val.csv
TEST_DATA=$DATA_DIR/labels/test_for_students.csv

cd $DATA_DIR

# train a mlp model
# and save the model in /models/soundnet-y_scns.mlp.model
python $BASE_DIR/train_mlp.py avg_pooling/y_scns 401 $TRAIN_DATA models/soundnet-y_scns.mlp.model


# test the model
# and save the result in /results/soundnet.csv
python $BASE_DIR/test_mlp.py models/soundnet-y_scns.mlp.model avg_pooling/y_scns 401 $TEST_DATA results/soundnet.csv