#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
run_name='Experiment'
rounds=5 epochs=500 patience=30 device='mps'
batch_size=1 learning_rate=0.001 decay=0.001
experiment=1 record=1 program_test=0 verbose=1 classification=0
dimensions="32"
datasets="cpu"
#densities="0.02 0.04"
train_sizes="5 50 100 500 900"
#train_sizes="500"
py_files="train_model"
# brp_nas
models="brp_nas"
patience=10
#models="mlp"
#models="lstm"
#models="gru"

for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
            for model in $models
            do
                for train_size in $train_sizes
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