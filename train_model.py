# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import pandas as pd
import torch

from tqdm import *

from baselines.brp_nas import BRP_NAS
from baselines.gru import GRU
from baselines.lstm import LSTM
from baselines.mlp import MLP
from data import experiment, DataModule
from baselines.gnn import GraphSAGEConv
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
from utils.utils import makedir
global log, args

torch.set_default_dtype(torch.float32)


class Model(torch.nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = args.dimension
        if args.model == 'brp_nas':
            self.model = BRP_NAS(args)

        elif args.model == 'gcn':
            self.model = GraphSAGEConv(6, self.hidden_size, args.order, self.args)

        elif args.model == 'mlp':
            self.model = MLP(6, self.hidden_size, 1, args)

        elif args.model == 'lstm':
            self.model = LSTM(54, self.hidden_size, 1, args)

        elif args.model == 'gru':
            self.model = GRU(54, self.hidden_size, 1, args)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, adjacency, features):
        y = self.model.forward(adjacency, features)
        return y.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if args.classification else 'max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader):
            graph, features, value = train_Batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features)
            loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        val_loss = 0.
        preds = []
        reals = []
        for valid_batch in tqdm(dataModule.valid_loader):
            graph, features, value = valid_batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features)
            val_loss += self.loss_function(pred, value.long())
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        self.scheduler.step(val_loss)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return valid_error

    def test_one_epoch(self, dataModule):
        preds = []
        reals = []
        for test_batch in tqdm(dataModule.test_loader):
            graph, features, value = test_batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return test_error


def RunOnce(args, runId, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)
    monitor = EarlyStopping(args)

    # Setup training tool
    model.setup_optimizer(args)
    train_time = []
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track_one_epoch(epoch, model, valid_error)
        train_time.append(time_cost)
        log.show_epoch_error(runId, epoch, monitor, epoch_loss, valid_error, train_time)
        plotter.append_epochs(valid_error)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule)
    log.show_test_error(runId, monitor, results, sum_time)

    # Save the best model parameters
    makedir('./checkpoints')
    model_path = f'./checkpoints/{args.model}_{args.seed}.pt'
    torch.save(monitor.best_model, model_path)
    # log.only_print(f'Model parameters saved to {model_path}')
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        plotter.reset_round()
        results = RunOnce(args, runId, log)
        plotter.append_round()
        for key in results:
            metrics[key].append(results[key])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)
        plotter.record_metric(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings(args)

    # logger plotter
    filename = f'{args.train_size}_r{args.dimension}'
    log = Logger(filename, args)
    plotter = MetricsPlotter(filename, args)
    args.log = log
    log(str(args.__dict__))

    # Run Experiment
    RunExperiments(log, args)

