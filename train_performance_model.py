import numpy as np
from .model.performance_preditor import PerformancePredictor
from .dataset.dataset import PerformanceDataset

batch_size = 4
dataset_path = "./dataset/merged_file.json"

train_set = PerformanceDataset(dataset_path)

net = PerformancePredictor()
