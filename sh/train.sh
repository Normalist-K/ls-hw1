#!/bin/bash
source ../config.sh
TRAIN_DATA=$DATA_DIR/labels/train_val.csv
TEST_DATA=$DATA_DIR/labels/test_for_students.csv


python train_mlp.py $DATA_DIR/avg_pooling/y_obj 1000 $TRAIN_DATA $DATA_DIR/models/soundnet-y_obj.mlp3.model --hidden_dim 1024 512 256
python test_mlp.py $DATA_DIR/models/soundnet-y_obj.mlp3.model $DATA_DIR/avg_pooling/y_obj 1000 $TEST_DATA results/soundnet-y_obj.mlp3.csv

python train_mlp.py $DATA_DIR/avg_pooling/y_scns 401 $TRAIN_DATA $DATA_DIR/models/soundnet-y_scns.mlp3.model --hidden_dim 1024 512 256
python test_mlp.py $DATA_DIR/models/soundnet-y_scns.mlp3.model $DATA_DIR/avg_pooling/y_scns 401 $TEST_DATA results/soundnet-y_scns.mlp3.csv


# python train_mlp.py $DATA_DIR/avg_pooling/y_obj 1000 $TRAIN_DATA $DATA_DIR/models/soundnet-y_obj.mlp.model
# python test_mlp.py $DATA_DIR/models/soundnet-y_obj.mlp.model $DATA_DIR/avg_pooling/y_obj 1000 $TEST_DATA results/soundnet-y_obj.mlp.csv

# python train_mlp.py $DATA_DIR/avg_pooling/y_scns 401 $TRAIN_DATA $DATA_DIR/models/soundnet-y_scns.mlp.model
# python test_mlp.py $DATA_DIR/models/soundnet-y_scns.mlp.model $DATA_DIR/avg_pooling/y_scns 401 $TEST_DATA results/soundnet-y_scns.mlp.csv


# ==== Feb 5 - 2:00 PM

# python train_mlp.py $DATA_DIR/avg_pooling/conv7 1024 $TRAIN_DATA models/soundnet-conv7.mlp.model --hidden_dim 1024 512 256
# python test_mlp.py models/soundnet-conv7.mlp.model $DATA_DIR/avg_pooling/conv7 1024 $TEST_DATA results/soundnet-conv7.mlp.csv

# ==== Feb 5 - 1:15 PM

# python train_mlp.py $DATA_DIR/avg_pooling/conv7 1024 $TRAIN_DATA models/soundnet-conv7.mlp.model
# python test_mlp.py models/soundnet-conv7.mlp.model $DATA_DIR/avg_pooling/conv7 1024 $TEST_DATA results/soundnet-conv7.mlp.csv

# python train_mlp.py $DATA_DIR/avg_pooling/y_obj 1024 $TRAIN_DATA models/soundnet-y_obj.mlp.model
# python test_mlp.py models/soundnet-y_obj.mlp.model $DATA_DIR/avg_pooling/y_obj 1024 $TEST_DATA results/soundnet-y_obj.mlp.csv

# python train_mlp.py $DATA_DIR/avg_pooling/y_scns 1024 $TRAIN_DATA models/soundnet-y_scns.mlp.model
# python test_mlp.py models/soundnet-y_scns.mlp.model $DATA_DIR/avg_pooling/y_scns 1024 $TEST_DATA results/soundnet-y_scns.mlp.csv

