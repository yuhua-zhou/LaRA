import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_loss_plot(y, title, labels, x_label="epoch", y_label="loss"):
    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        plt.plot(x, y[i], label=labels[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def draw_heatmap():
    # 定义行名和列名
    row_names = ['llama7b-0.5', 'llama7b-0.25', 'llama7b-0.3', 'llama7b-0.2',
                 'vicuna7b-0.5', 'vicuna7b-0.25', 'vicuna7b-0.3', 'vicuna7b-0.2']

    # 定义偶数列表
    even_numbers = [i for i in range(2, 17, 2)]

    # 生成随机数据矩阵，数值为偶数
    data = np.random.choice(even_numbers, size=(8, 32))

    # 创建一个热力图
    plt.figure(figsize=(32, 8))
    sns.heatmap(data, cmap='YlGnBu', yticklabels=row_names, annot=True)

    # 显示图表
    plt.show()
