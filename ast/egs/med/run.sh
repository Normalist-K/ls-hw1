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
base_exp_dir=$DATA_DIR/exp/best

# python ./prep_med.py

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $base_exp_dir

for((fold=0;fold<=4;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=/shared/youngkim/dataset/datafiles/med_train_data_${fold}.json
  te_data=/shared/youngkim/dataset/datafiles/med_eval_data_${fold}.json
  ev_data=/shared/youngkim/dataset/datafiles/med_test_data.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --data-eval ${ev_data} --n_class 15 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done

python ./get_med_result.py --exp_path ${base_exp_dir}