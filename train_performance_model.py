import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.dataset import PerformanceDataset
from model.loss import WeightedMSELoss
from model.performance_preditor import PerformancePredictor
from utils.plot import draw_loss_plot

# hyper-parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_epoch = 100
batch_size = 64
lr = 1e-3
dataset_path = "dataset/merged_file_v2.json"
prune_path = "./rankadaptor/prune_log/local/"
metrics = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']

# create dataset
train_set = PerformanceDataset(dataset_path, prune_path, "train", augment=5000)
train_set.statistics()

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = PerformanceDataset(dataset_path, prune_path, "test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# create model
net = PerformancePredictor(input_size=64, hidden_size=128).double().to(device)
criterion = WeightedMSELoss()
# criterion = WeightedLogCoshLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5)
# lmbda = lambda epoch: 0.99 ** epoch
# scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

n_metrics = len(metrics)
train_loss_list = []  # train loss for each metric
test_loss_list = []  # test loss for each metric
total_train_test_loss = [[], []]  # total train and test loss

for epoch in range(num_epoch):
    net.train()
    print(f"learning rate = {scheduler.get_last_lr()}")

    # 记录训练损失
    running_loss = [0.0] * n_metrics
    total_train_loss = 0.0

    for j, batch in enumerate(train_loader):
        (layer_info, rank_list, prune_list, performance) = batch
        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        output = net(layer_info, rank_list, prune_list)
        total_loss = criterion(output, performance)
        total_train_loss += 100 * total_loss.item()

        # (batch, metrics) -> (metrics, batch)
        output = output.transpose(0, 1)
        performance = performance.transpose(0, 1)

        for i in range(n_metrics):
            loss = criterion(output[i], performance[i])
            running_loss[i] += 100 * loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / len(train_set) / n_metrics
    total_train_test_loss[0].append(total_train_loss)

    running_loss = [l / len(train_set) for l in running_loss]
    train_loss_list.append(running_loss)

    print('%s: epoch: %d, train loss: %.6f \r'
          % (datetime.datetime.now(), epoch, total_train_loss))

    # ------------------------------ test ----------------------------

    net.eval()
    with torch.no_grad():

        # 记录测试损失
        testing_loss = [0.0] * n_metrics
        total_test_loss = 0.0

        for j, batch in enumerate(test_loader):
            (layer_info, rank_list, prune_list, performance) = batch
            layer_info, rank_list, prune_list, performance = (
                Variable(layer_info).to(device), Variable(rank_list).to(device),
                Variable(prune_list).to(device), Variable(performance).to(device))

            output = net(layer_info, rank_list, prune_list)
            total_loss = criterion(output, performance)
            total_test_loss += 100 * total_loss.item()

            # (batch, metrics) -> (metrics, batch)
            output = output.transpose(0, 1)
            performance = performance.transpose(0, 1)

            for i in range(n_metrics):
                loss = criterion(output[i], performance[i])
                testing_loss[i] += 100 * loss.item()

        total_test_loss = total_test_loss / len(test_set) / n_metrics
        total_train_test_loss[1].append(total_test_loss)

        testing_loss = [l / len(test_set) for l in testing_loss]
        test_loss_list.append(testing_loss)

        print('%s: epoch: %d, test loss: %.6f \r'
              % (datetime.datetime.now(), epoch, total_test_loss))

    print("-" * 64)
    scheduler.step()

train_loss_list = np.array(train_loss_list).transpose((1, 0))
test_loss_list = np.array(test_loss_list).transpose((1, 0))
total_train_test_loss = np.array(total_train_test_loss)

draw_loss_plot(y=train_loss_list, title="training loss", labels=metrics)
draw_loss_plot(y=test_loss_list, title="testing loss", labels=metrics)
draw_loss_plot(y=total_train_test_loss, title="total loss", labels=["total_train", "total_test"])

torch.save(net.state_dict(), "./output/performance_weights.pth")
