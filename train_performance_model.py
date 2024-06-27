import math
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
import time
import datetime

# hyper-parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_epoch = 500
batch_size = 16
lr = 1e-4
dataset_path = "./dataset/merged_file_revise.json"

# create dataset
train_set = PerformanceDataset(dataset_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# create model
net = PerformancePredictor().double().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
lmbda = lambda epoch: 0.95
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)


def draw_loss_plot(x, y):
    plt.plot(x, y, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


loss_list = []
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        (layer_info, rank_list, prune_list, performance) = batch
        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        output = net(layer_info, rank_list, prune_list)

        loss = criterion(output, performance)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # scheduler.step()

    print('%s: epoch: %d, loss: %.5f \r' % (datetime.datetime.now(), epoch, running_loss))
    loss_list.append(running_loss)

draw_loss_plot(range(num_epoch), loss_list)
torch.save(net.state_dict(), "./output/performance_weights.pth")


# a = [-0.0023, -0.0580, -0.0099, 0.1217, -0.1391, 0.0885, 0.0452]
# b = [0.6768, 0.3805, 0.6385, 0.4080, 0.5651, 0.7709, 0.6732]
#
# c = [0.0046, 0.0687, 0.1304, 0.0999, -0.0421, 0.0419, 0.0800]
# d = [0.6162, 0.3746, 0.6267, 0.4000, 0.6590, 0.7682, 0.6835]
#
# loss = 0.0
#
# for i in range(len(a)):
#     loss += math.pow(a[i] - b[i], 2) + math.pow(c[i] - d[i], 2)
#
# print(loss, loss / 7, loss / 14)
#
# loss = criterion(torch.tensor([a, c]), torch.tensor([b, d]))
# print(loss)
