#!/bin/bash

set -x
source ./venvast/bin/activate
export TORCH_HOME=./pretrained_models

model=ast
dataset=cough
imagenetpretrain=False
audiosetpretrain=False

bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi

freqm=24
timem=96
mixup=0
epoch=2
batch_size=30
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

for((fold=0;fold<=4;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./cough_train_${fold}.json
  te_data=./cough_eval_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ./src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./newLabel_OnlyTwo.csv --n_class 2 \
  --lr $lr -w 0 --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done


python ./get_esc_result.py --exp_path ${base_exp_dir}

cat ./${base_exp_dir}/fold*/predictions/predictions_2.csv > predictions.txt

## now open step.2.calculate.ROC.R script to calculate all the metric

Rscript step.2.calculate.ROC.R







