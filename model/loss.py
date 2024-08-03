import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.weight = weight

    def forward(self, input, target):
        loss = self.mse_loss(input, target)
        loss = torch.sqrt(loss)

        if self.weight is not None:
            loss = loss * self.weight.expand_as(loss)

        return loss.sum()


class WeightedLogCoshLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        loss = torch.log(torch.cosh(input - target))

        if self.weights is not None:
            loss = self.weights * loss

        return loss.sum()


if __name__ == '__main__':
    # 示例使用
    # weight = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    #
    # input = torch.tensor([[1, 2, 3, 5], [1, 2, 3, 5]], dtype=torch.float)
    # target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float)
    #
    # criterion = WeightedMSELoss(weight, l1_lambda=0.1)
    # loss = criterion(input, target)
    # print(loss)
    #
    # criterion = WeightedLogCoshLoss(weight, l1_lambda=0.1)
    # loss = criterion(input, target)
    # print(loss)

    input = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.float)
    target = torch.tensor([[1, 3], [1, 4], [2, 3]], dtype=torch.float)
    criterion = WeightedMSELoss()
    loss = criterion(input, target)
    print(loss.item())
