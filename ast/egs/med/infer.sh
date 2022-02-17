#!/bin/bash
source ../../../config.sh

model=ast
dataset=med
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=48
timem=192
mixup=0
epoch=10
batch_size=24
fstride=10
tstride=10
base_exp_dir=$DATA_DIR/infer


if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $base_exp_dir

exp_dir=${base_exp_dir}

tr_data=$DATA_DIR/datafiles/med_train_data.json
te_data=$DATA_DIR/datafiles/med_eval_data_0.json
ev_data=$DATA_DIR/datafiles/med_test_data.json

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/infer.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--data-eval ${ev_data} --n_class 15 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
