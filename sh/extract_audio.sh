#!/bin/bash
source ./config.sh

cd $DATA_DIR
mkdir wav/ mp3/
for file in videos/*; do; 
    filename=$(basename $file .mp4); 
    ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; 
    ffmpeg -y -i $file -ac 1 -f mp3 -ar 22050 mp3/${filename}.mp3;
done