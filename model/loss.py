import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.weight = weight

    def forward(self, input, target):
        loss = self.mse_loss(input, target)
        if self.weight is not None:
            loss = loss * self.weight.expand_as(loss)

        return loss.mean()

if __name__ == '__main__':
    # 示例使用
    weight = torch.tensor([1, 1, 1, 1], dtype=torch.float)

    input = torch.tensor([[1, 2, 3, 5], [1, 2, 3, 5]], dtype=torch.float)
    target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float)

    criterion = WeightedMSELoss(weight)
    loss = criterion(input, target)
    print(loss)
