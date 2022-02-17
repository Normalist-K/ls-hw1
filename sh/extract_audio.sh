#!/bin/bash
source ./config.sh

cd $DATA_DIR

mkdir wav/ mp3/ wav_16k/
for file in videos/*; do
    filename=$(basename $file .mp4); 
    ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; 
    ffmpeg -y -i $file -ac 1 -f mp3 -ar 22050 mp3/${filename}.mp3;
    ffmpeg -y -i $file -ac 1 -f wav -ar 16000 wav_16k/${filename}.wav; 
done