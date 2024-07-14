#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
experiment=1
run_name='Experiment'
rounds=5 epochs=500 patience=10 device='cpu'
batch_size=32 learning_rate=0.001 decay=0.001
record=1 program_test=0 verbose=1 classification=0
dimensions="50"
datasets="cpu"
train_sizes="50 100 200 400 900"

py_files="train_model"
## models here ##
#models="brp_nas"
models="mlp lstm gru birnn"

for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
						for train_size in $train_sizes
            do
            		for model in $models
                do
                    python ./$py_file.py \
                          --device $device \
                          --logger $run_name \
                          --rounds $rounds \
                          --train_size $train_size \
                          --dataset $dataset \
                          --model $model \
                          --bs $batch_size \
                          --epochs $epochs \
                          --patience $patience \
                          --bs $batch_size \
                          --lr $learning_rate \
                          --decay $decay \
                          --program_test $program_test \
                          --dimension $dim \
                          --experiment $experiment \
                          --record $record \
                          --verbose $verbose \
                          --classification $classification

                done
            done
        done
    done
done