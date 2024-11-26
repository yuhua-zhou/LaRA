import datetime
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.dataset import PerformanceDataset
from model.loss import WeightedMSELoss
from model.performance.performance_preditor import PerformancePredictor
from tqdm import tqdm


def main(weight_name):
    # hyper-parameters
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_epoch = 50
    batch_size = 128
    lr = 1e-3
    dataset_path = "dataset/merged_file_v3.json"
    prune_path = "./rankadaptor/prune_log/local/"
    metrics = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']

    # create dataset
    train_set = PerformanceDataset(dataset_path, prune_path, "train", augment=3000)
    weight = train_set.statistics()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = PerformanceDataset(dataset_path, prune_path, "test", augment=100)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create model
    net = PerformancePredictor(input_size=128, hidden_size=128).double().to(device)
    criterion = WeightedMSELoss(weight=weight[weight_name].to(torch.float64).to(device))
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    n_metrics = len(metrics)
    total_train_test_loss = []  # total train and test loss

    for epoch in tqdm(range(num_epoch)):
        net.train()

        for j, batch in enumerate(train_loader):
            (layer_info, rank_list, prune_list, performance) = batch
            layer_info, rank_list, prune_list, performance = (
                Variable(layer_info).to(device), Variable(rank_list).to(device),
                Variable(prune_list).to(device), Variable(performance).to(device))

            output = net(layer_info, rank_list, prune_list)
            total_loss = criterion(output, performance)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # ------------------------------ test ----------------------------

        net.eval()
        with torch.no_grad():

            # 记录测试损失
            total_test_loss = 0.0

            for j, batch in enumerate(test_loader):
                (layer_info, rank_list, prune_list, performance) = batch
                layer_info, rank_list, prune_list, performance = (
                    Variable(layer_info).to(device), Variable(rank_list).to(device),
                    Variable(prune_list).to(device), Variable(performance).to(device))

                output = net(layer_info, rank_list, prune_list)
                total_loss = criterion(output, performance)
                total_test_loss += 100 * total_loss.item()

            total_test_loss = total_test_loss / len(test_set) / n_metrics
            total_train_test_loss.append(total_test_loss)

        scheduler.step()

    with open("./output/ablation_performance/" + weight_name + ".json", "w+") as file:
        file.write(json.dumps(total_train_test_loss))
        print(f"save ./output/ablation_performance/{weight_name}.json successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_name', type=str, default="equal", help='weight name')
    args = parser.parse_args()
    main(args.weight_name)
