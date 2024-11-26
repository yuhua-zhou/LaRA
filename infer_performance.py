from model.loss import WeightedMSELoss
from model.performance.performance_preditor import PerformancePredictor
from dataset.dataset import PerformanceDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

# hyper-parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset_path = "dataset/merged_file_v2.json"
prune_path = "./rankadaptor/prune_log/local/"
batch_size = 1
metrics = ['arc_easy', 'arc_challenge', 'winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag']

test_set = PerformanceDataset(dataset_path, prune_path, "test")
test_set.statistics()
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# create model
net = PerformancePredictor(input_size=64, hidden_size=128).double().to(device)
net.load_state_dict(torch.load("./output/performance_weights_svd_3090.pth"))
criterion = WeightedMSELoss()

net.eval()
with torch.no_grad():
    n_metrics = len(metrics)
    for i, batch in enumerate(test_loader):
        (layer_info, rank_list, prune_list, performance) = batch
        layer_info, rank_list, prune_list, performance = (
            Variable(layer_info).to(device), Variable(rank_list).to(device),
            Variable(prune_list).to(device), Variable(performance).to(device))

        output = net(layer_info, rank_list, prune_list)
        total_loss = criterion(output, performance)

        # (batch, metrics) -> (metrics, batch)
        output = output.transpose(0, 1)
        performance = performance.transpose(0, 1)

        for i in range(n_metrics):
            loss = criterion(output[i], performance[i])
            print(
                f"{metrics[i]}: {output[i].detach().cpu().numpy()}, {performance[i].detach().cpu().numpy()}, {loss.item()}")

        print(f"total_loss: {total_loss.item()}\n\n")
