#!/bin/bash
source ./config.sh
TRAIN_DATA=$DATA_DIR/labels/kfold_df.csv
TEST_DATA=$DATA_DIR/labels/test_for_students.csv

cd $DATA_DIR

# train a svm model
# and save the model in /models/mfcc-50.svm.multiclass.model
python $BASE_DIR/train_svm_multiclass.py bof/ 50 $TRAIN_DATA models/mfcc-50.svm.multiclass.model

# test the model
# and save the result in /results/mfcc.csv
mkdir results
$ python $BASE_DIR/test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 $TEST_DATA results/mfcc.csv
