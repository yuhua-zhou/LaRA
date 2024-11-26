import datetime
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.dataset import PerformanceDataset
from model.nas.nas_model import PolicyNetwork, NASAgent
from utils.plot import draw_loss_plot

device = "cuda:0" if torch.cuda.is_available() else "cpu"

action_space = [2, 4, 6, 8, 10, 12, 14, 16]
action_map = {float(a): i for i, a in enumerate(action_space)}
embed_size = 128
num_epoch = 1000
batch_size = 128
lr = 1e-3
random.seed(20241126)

dataset_path = "dataset/merged_file_v4.json"
prune_path = "./rankadaptor/prune_log/local/"
train_set = PerformanceDataset(dataset_path, prune_path, "train", augment=0)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

policy_net = PolicyNetwork(embed_size, 128, len(action_space)).double().to(device)
agent = NASAgent(policy_net)

# 定义优化器
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-4)

train_loss = []
train_entropy = []

for epoch in range(num_epoch):
    policy_net.train()
    print(f"learning rate = {scheduler.get_last_lr()}")

    running_loss = []
    running_entropy = []
    for j, batch in enumerate(train_loader):
        (layer_info, rank_list, prune_list, performance) = batch

        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        reward = torch.mean(performance, dim=1)
        reward = reward.unsqueeze(1)

        rank_list = rank_list.squeeze(2).tolist()

        rank_indices = torch.tensor([
            [[action_map[rr]] for rr in r]
            for r in rank_list
        ], device=device)

        logits = policy_net(layer_info, prune_list)
        probs = F.softmax(logits / agent.temperature, dim=2)
        probs = torch.gather(probs, dim=2, index=rank_indices)
        probs = probs.squeeze(2)
        log_probs = torch.log(probs)

        loss = agent.update_policy(log_probs, reward)
        entropy = -(probs * torch.log(probs)).sum()  # 策略熵

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        running_entropy.append(entropy.detach().item())

    mean_loss = sum(running_loss) / len(running_loss)
    mean_entropy = sum(running_entropy) / len(running_entropy)
    print("%s: epoch: %d, loss: %s, entropy: %s" % (datetime.datetime.now(), epoch, mean_loss, mean_entropy))

    train_loss.append(mean_loss)
    train_entropy.append(mean_entropy)
    scheduler.step()

draw_loss_plot(y=[train_loss], title="training loss", labels=["total loss"])
draw_loss_plot(y=[train_entropy], title="training entropy", labels=["total entropy"])
torch.save(policy_net.state_dict(), "./output/nas_weights_pretrain.pth")
