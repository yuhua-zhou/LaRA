import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR
from model.performance_preditor import PerformancePredictor
from dataset.dataset import PerformanceDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.plot import draw_loss_plot
import datetime

# hyper-parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_epoch = 500
batch_size = 16
lr = 1e-5
dataset_path = "dataset/merged_file_v1.json"
prune_path = "./rankadaptor/prune_log/local/"
metrics = ['arc_easy', 'arc_challenge', 'winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag']

# create dataset
train_set = PerformanceDataset(dataset_path, prune_path, "train")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = PerformanceDataset(dataset_path, prune_path, "test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# create model
net = PerformancePredictor().double().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
lmbda = lambda epoch: 0.95
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

n_metrics = len(metrics)
train_loss_list = []
test_loss_list = []

for epoch in range(num_epoch):

    # 记录训练损失
    running_loss = [0.0] * n_metrics
    for i, batch in enumerate(train_loader):
        (layer_info, rank_list, prune_list, performance) = batch

        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        output = net(layer_info, rank_list, prune_list)
        total_loss = criterion(output, performance)

        total_loss.backward()
        optimizer.step()

        # (batch, metrichs) -> (metrics, batch)
        output = output.transpose(0, 1)
        performance = performance.transpose(0, 1)

        for i in range(n_metrics):
            loss = criterion(output[i], performance[i])
            running_loss[i] += loss.item()

        # scheduler.step()

    print('%s: epoch: %d, train loss: %.5f \r'
          % (datetime.datetime.now(), epoch, np.sum(np.array(running_loss))))
    train_loss_list.append(running_loss)

    # 记录测试损失
    testing_loss = [0.0] * n_metrics
    for i, batch in enumerate(test_loader):
        (layer_info, rank_list, prune_list, performance) = batch
        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        output = net(layer_info, rank_list, prune_list)

        # (batch, metrichs) -> (metrics, batch)
        output = output.transpose(0, 1)
        performance = performance.transpose(0, 1)

        for i in range(n_metrics):
            loss = criterion(output[i], performance[i])
            testing_loss[i] += loss.item()

    print('%s: epoch: %d, test loss: %.5f \r'
          % (datetime.datetime.now(), epoch, np.sum(np.array(testing_loss))))
    test_loss_list.append(testing_loss)

train_loss_list = np.array(train_loss_list).transpose((1, 0))
test_loss_list = np.array(test_loss_list).transpose((1, 0))

draw_loss_plot(y=train_loss_list, title="training loss", labels=metrics)
draw_loss_plot(y=test_loss_list, title="testing loss", labels=metrics)

torch.save(net.state_dict(), "./output/performance_weights.pth")
