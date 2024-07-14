#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --llm 1
python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --llm 0
#python train_model.py --config_path ./exper_config.py --exp_name GCNConfig
#python train_model.py --config_path ./exper_config.py --exp_name BrpNASConfig
#python train_model.py --config_path ./exper_config.py --exp_name MLPConfig
#python train_model.py --config_path ./exper_config.py --exp_name LSTMConfig
#python train_model.py --config_path ./exper_config.py --exp_name GRUConfig
#python train_model.py --config_path ./exper_config.py --exp_name BiRnnConfig
