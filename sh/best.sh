#!/bin/bash
source ./config.sh

cd $BASE_DIR/ast/egs/med

# create .json format dataset
# and save /$DATA_DIR/datafiles/*.json
python prep_med.py $BASE_DIR $DATA_DIR

# train & validation (5fold cross validation) & inference
bash run.sh

# save best.csv
python gen_submission.py $BASE_DIR $DATA_DIR 
