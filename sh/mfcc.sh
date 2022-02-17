#!/bin/bash
source ./config.sh

cd $DATA_DIR

# .wav -> mfcc
mkdir mfcc/
for file in wav/*;do 
    filename=$(basename $file .wav);
    $BASE_DIR/tools/opensmile-3.0.1-linux-x64/bin/SMILExtract -C $BASE_DIR/config/MFCC12_0_D_A.conf -I ${file} -O mfcc/${filename}.mfcc.csv;
done

# select frames using 20% sampling
# and save the result in /mfcc/selected.mfcc.csv
python $BASE_DIR/select_frames.py --input_path labels/train_val.csv --ratio 0.2 --output_path mfcc/selected.mfcc.csv --mfcc_dir mfcc/

# train K-Means clustering (k=50) => CODEBOOK
mkdir models/
python $BASE_DIR/train_kmeans.py -i mfcc/selected.mfcc.csv -k 50 -o models/kmeans.50.model

# feature extraction
# and save the results in /bof
mkdir bof/
$ ls videos/ | while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
$ python $BASE_DIR/get_bof.py models/kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/