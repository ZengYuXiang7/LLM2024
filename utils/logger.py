# coding : utf-8
# Author : yuxiang Zeng
# 日志
import logging
import pickle
import sys
import time
import numpy as np
import platform

from utils.utils import makedir

class Logger:
    def __init__(self, filename, exper_datail, args):
        self.args = args
        self.exper_datail = exper_datail
        self.fileroot = f'./results/{args.model}/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '/log/'
        self.filename = filename
        makedir(self.fileroot)
        exper_time = time.strftime('%H_%M_%S', time.localtime(time.time())) + '_'
        exper_filename = self.fileroot + exper_time + self.filename
        logging.basicConfig(level=logging.INFO, filename=f"{exper_filename}.log", filemode='w')
        self.logger = logging.getLogger(self.args.model)

    def save_result(self, metrics):
        args = self.args
        makedir('./results/metrics/')
        address = f'./results/metrics/' + self.filename
        for key in metrics:
            pickle.dump(np.mean(metrics[key]), open(address + key + 'mean.pkl', 'wb'))
            pickle.dump(np.std(metrics[key]), open(address + key + 'std.pkl', 'wb'))

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        self.only_print(string)
        self.logger.info(final_string)

    def __call__(self, string):
        if self.args.verbose:
            self.log(string)

    def only_print(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        green_string = f"\033[1;38;2;151;200;129m{time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time()))}\033[0m" + f"\033[1m{string}\033[0m"
        print(green_string)

    def show_epoch_error(self, runId, epoch, monitor, epoch_loss, result_error, train_time):
        if self.args.verbose and epoch % self.args.verbose == 0 and not self.args.program_test:
            self.only_print(f"\033[1;38;2;151;200;129m{self.exper_datail}\033[0m")
            self.only_print(f'Best NMAE Epoch {monitor.best_epoch} = {monitor.best_score * -1:.4f}  now = {(epoch - monitor.best_epoch):d}')
            if self.args.classification:
                self.only_print(f"Round={runId + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} vAcc={result_error['Acc']:.4f} vF1={result_error['F1']:.4f} vPrecision={result_error['P']:.4f} vRecall={result_error['Recall']:.4f} time={sum(train_time):.1f} s ")
            else:
                self.only_print(f"Round={runId + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} vMAE={result_error['MAE']:.4f} vRMSE={result_error['RMSE']:.4f} vNMAE={result_error['NMAE']:.4f} vNRMSE={result_error['NRMSE']:.4f} time={sum(train_time):.1f} s ")
                self.only_print(f"Acc = [1%={result_error['Acc_1']:.4f}, 5%={result_error['Acc_5']:.4f}, 10%={result_error['Acc_10']:.4f}] ")

    def show_test_error(self, runId, monitor, results, sum_time):
        if self.args.classification:
            self(f'Round={runId + 1} BestEpoch={monitor.best_epoch:3d} Acc={results["Acc"]:.4f} F1={results["F1"]:.4f} Precision={results["P"]:.4f} Recall={results["Recall"]:.4f} Training_time={sum_time:.1f} s \n')
        else:
            self(f'Round={runId + 1} BestEpoch={monitor.best_epoch:3d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s ')
            self(f"Acc = [1%={results['Acc_1']:.4f}, 5%={results['Acc_5']:.4f}, 10%={results['Acc_10']:.4f}] ")
        print()
