import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_loss_plot(y, title, labels, x_label="epoch", y_label="loss", save_name=None):
    plt.rcParams.update({'font.size': 14})  # 设置默认字体大小

    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        plt.plot(x, y[i], label=labels[i])

    plt.title(title, fontsize=18)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.legend()

    if save_name is not None:
        plt.savefig(f"{save_name}.pdf")

    plt.show()


def draw_heatmap():
    # 定义行名和列名
    row_names = ['llama7b-0.20', 'llama7b-0.25', 'llama7b-0.30', 'llama7b-0.50',
                 'vicuna7b-0.20', 'vicuna7b-0.25', 'vicuna7b-0.30', 'vicuna7b-0.50']

    optimal_ranks = [
        [8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 4, 2, 2, 2, 4, 2, 6, 2, 8, 4, 4, 8, 2, 12, 8, 2, 10, 10, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 4, 2, 2, 2, 4, 2, 6, 2, 8, 4, 4, 8, 2, 12, 8, 2, 10, 10, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 2, 12, 10, 2, 4, 2, 2, 4, 16, 2, 16, 16, 2, 2, 2, 2, 10, 2, 10, 14, 12, 2, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 2, 2, 6, 6, 2, 2, 2, 4, 8, 2, 4, 16, 6, 10, 4, 8, 10, 2, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 4, 4, 10, 4, 4, 14, 2, 10, 16, 16, 4, 2, 4, 6, 10, 14, 12, 10, 4, 10, 12, 4, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 12, 4, 10, 4, 4, 10, 14, 2, 16, 4, 10, 16, 2, 2, 2, 2, 14, 2, 2, 14, 14, 14, 10, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 10, 10, 2, 14, 2, 8, 2, 8, 12, 2, 2, 2, 2, 8, 2, 8, 8, 12, 8, 2, 8, 8, 10, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 4, 10, 2, 2, 16, 2, 2, 6, 12, 4, 4, 4, 4, 4, 2, 8, 4, 6, 6, 4, 4, 2, 8, 10, 8, 8, 8],
    ]

    for i in range(len(row_names)):
        print(row_names[i], ":", sum(optimal_ranks[i]))

    # 创建一个热力图
    plt.figure(figsize=(32, 8))
    ax = sns.heatmap(optimal_ranks, cmap='YlGnBu', yticklabels=row_names, annot=True,
                     annot_kws={"size": 20},
                     cbar_kws={"pad": 0.01}
                     )

    # 设置x轴和y轴的标签文本大小
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴刻度标签的字体大小
    ax.tick_params(axis='y', labelsize=16)  # 设置y轴刻度标签的字体大小

    # 设置x轴和y轴的标题文本大小
    ax.set_xlabel('Layer Index', fontsize=18)
    ax.set_ylabel('Models', fontsize=18)

    plt.savefig("heatmap.pdf", format='pdf')

    # 显示图表
    plt.show()


def draw_evaluation_line(model_name):
    metrics = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']
    name = model_name.split("/")[-1]

    performance = [[], [], [], [], [], [], []]

    for i in range(200, 1600, 200):
        file_path = model_name + "_" + str(i) + ".json"
        with open(file_path, "r") as file:
            line = json.load(file)
            line = line["results"]

            for j in range(len(metrics)):
                metric = metrics[j]
                performance[j].append(line[metric]["acc"])

    performance = np.array(performance)

    for i in range(len(metrics)):
        x = range(200, 1600, 200)
        plt.plot(x, performance[i], label=metrics[i])

    plt.title(f"evaluation of {name}")
    plt.xlabel("value")
    plt.ylabel("step")
    plt.legend()
    plt.show()


def draw_performance_ablation():
    path = "../output/ablation_performance/"

    names = []
    data = []

    for file in os.listdir(path):
        if file.endswith(".json"):
            names.append(file.split(".")[0])
            line = json.load(open(path + file, "r"))
            data.append(line)

    print(names)
    print(data)

    draw_loss_plot(y=data, labels=names, title="Testing Loss Across Different Weight", save_name="ablation_performance")


def draw_adalora():
    # 设置全局默认字体大小
    plt.rcParams['font.size'] = 16

    # Parameters for the simulation
    t_init = 100  # Warmup steps
    t_final = 500  # Steps where pruning ends

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    steps = range(0, 900, 100)
    memory = [12, 12, 11, 10, 9, 8, 8, 8, 8]

    # 绘制第二个子图（第2行）
    ax1.plot(steps, memory, label='AdaLoRA', color='blue', linewidth=2)
    ax1.plot(steps, [8, 8, 8, 8, 8, 8, 8, 8, 8], label='Uniform Assignment', color='skyblue', linewidth=2)
    # ax1.plot(steps, [8] * 9, label='Random Rank', color='orange')

    ax1.axvline(x=t_init, color="green", linestyle="--", label="t_init (Warmup Ends)", alpha=0.7)
    ax1.axvline(x=t_final, color="red", linestyle="--", label="t_final (Pruning Ends)", alpha=0.7)

    ax1.set_title('(A) Memory Usage Trend During Training (Rank value)', fontsize=20)
    ax1.set_xlabel('Steps', fontsize=18)
    ax1.set_ylabel('Memory Usage', fontsize=18)
    ax1.set_ylim(2, 16)  # 设置 Y 轴的范围为 20 到 80
    ax1.grid(True, linestyle='--', color="gray", alpha=0.5)
    ax1.legend()

    # ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------

    # 数据
    metrics = ["boolq", "piqa", "hellaswag", "winogrande", "arc-e", "arc-c", "obqa"]
    adalora = [55.35, 63.60, 39.79, 53.28, 38.43, 27.82, 37.20]
    random_rank = [55.84, 69.42, 45.77, 52.80, 45.92, 27.90, 34.60]

    # 设置柱状图位置
    x = np.arange(len(metrics))  # 位置
    width = 0.40  # 柱子的宽度

    # 绘制柱状图
    bars1 = ax2.bar(x - width / 2, adalora, width, label='AdaLoRA', color='#DEECF9', edgecolor='black', linewidth=1,
                    zorder=2)
    bars2 = ax2.bar(x + width / 2, random_rank, width, label='Random Rank', color='#F7CCAD', edgecolor='black',
                    linewidth=1, zorder=2)

    # 添加标签、标题和自定义刻度
    ax2.set_xlabel('Metrics', fontsize=18)
    ax2.set_ylabel('Accuracy', fontsize=18)
    ax2.set_title('(B) Comparison of AdaLoRA and Random Rank', fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=-15, ha='center')
    ax2.grid(True, linestyle='--', color="gray", alpha=0.5)
    ax2.legend()

    plt.tight_layout()

    plt.savefig("../figures/adalora.pdf")

    plt.show()


def draw_extra_metrics():
    # 设置全局默认字体大小
    plt.rcParams['font.size'] = 16

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # ----------------------------------------------------------------------------------------------------
    # 11.83，36.39
    metrics = ["Llama7B-50%", "Vicuna7B-50%"]
    lora = [8.80, 8.90]
    adalora = [8.43, 8.57]
    lara = [9.05, 8.95]

    # 设置柱状图位置
    x = np.arange(len(metrics))  # 位置
    width = 0.20  # 柱子的宽度

    # 绘制柱状图
    ax1.bar(x, lora, width, label='LoRA_r8', color='#DEECF9', edgecolor='black', linewidth=1, zorder=2)
    ax1.bar(x + width, adalora, width, label='Adalora', color='#F7CCAD', edgecolor='black', linewidth=1, zorder=2)
    ax1.bar(x + 2 * width, lara, width, label='LaRA', color='#9EC6BF', edgecolor='black', linewidth=1, zorder=2)

    # 添加标签、标题和自定义刻度
    ax1.set_xlabel('Models', fontsize=18)
    ax1.set_ylabel('Performance', fontsize=18)
    ax1.set_title('(A) GSM8K Benchmark', fontsize=20)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metrics, rotation=0, ha='center')
    ax1.grid(True, linestyle='--', color="gray", alpha=0.5)
    ax1.legend()

    # ----------------------------------------------------

    # 数据
    metrics = ["Llama7B-50%", "Vicuna7B-50%"]
    lora = [27.04, 27.38]
    adalora = [25.94, 26.36]
    lara = [27.84, 27.54]

    # 设置柱状图位置
    x = np.arange(len(metrics))  # 位置
    width = 0.20  # 柱子的宽度

    # 绘制柱状图
    ax2.bar(x, lora, width, label='LoRA_r8', color='#DEECF9', edgecolor='black', linewidth=1, zorder=2)
    ax2.bar(x + width, adalora, width, label='Adalora', color='#F7CCAD', edgecolor='black', linewidth=1, zorder=2)
    ax2.bar(x + 2 * width, lara, width, label='LaRA', color='#9EC6BF', edgecolor='black', linewidth=1, zorder=2)

    # 添加标签、标题和自定义刻度
    ax2.set_xlabel('Models', fontsize=18)
    ax2.set_ylabel('Performance', fontsize=18)
    ax2.set_title('(B) MMLU Benchmark', fontsize=20)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metrics, rotation=0, ha='center')
    ax2.grid(True, linestyle='--', color="gray", alpha=0.5)
    ax2.legend()

    plt.tight_layout()

    plt.savefig("../figures/extra_metric.pdf")

    plt.show()


def draw_training_and_testing():
    plt.rcParams.update({'font.size': 14})  # 设置默认字体大小

    # 读取数据
    performance = json.load(open("./performance.json", "r"))
    performance_training_loss = performance["training_loss"]
    performance_testing_loss = performance["testing_loss"]
    nas = json.load(open("./nas.json", "r"))
    nas_training_loss = nas["training_loss"]
    nas_training_entropy = nas["training_entropy"]

    # 创建图形和坐标轴
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # 0, 0
    labels = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']
    y = performance_training_loss

    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        ax1.plot(x, y[i], label=labels[i])

    ax1.set_title("(A) Training Loss", fontsize=18)
    ax1.set_xlabel("epoch", fontsize=15)
    ax1.set_ylabel("loss", fontsize=15)
    ax1.legend()

    # 0, 1
    y = performance_testing_loss

    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        ax2.plot(x, y[i], label=labels[i])

    ax2.set_title("(B) Testing Loss", fontsize=18)
    ax2.set_xlabel("epoch", fontsize=15)
    ax2.set_ylabel("loss", fontsize=15)
    ax2.legend()

    # 1, 0
    y = nas_training_loss
    labels = ["training loss"]

    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        ax3.plot(x, y[i], label=labels[i])

    ax3.set_title("(C) Training Loss", fontsize=18)
    ax3.set_xlabel("epoch", fontsize=15)
    ax3.set_ylabel("loss", fontsize=15)
    ax3.legend()

    # 1, 1
    y = nas_training_entropy
    labels = ["training entropy"]

    n = len(y)
    for i in range(n):
        x = range(len(y[i]))
        ax4.plot(x, y[i], label=labels[i])

    ax4.set_title("(D) Training Entropy", fontsize=18)
    ax4.set_xlabel("epoch", fontsize=15)
    ax4.set_ylabel("loss", fontsize=15)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("../figures/training_loss.pdf")

    plt.show()


if __name__ == "__main__":
    # base_path = "../rankadaptor/results/r8/"
    # model_names = ["llama7b-0.20_r8", "llama7b-0.50_r8", "vicuna7b-0.50_r8"]
    #
    # for model_name in model_names:
    #     draw_evaluation_line(base_path + model_name)

    # draw_heatmap()
    # draw_performance_ablation()
    # draw_adalora()
    draw_extra_metrics()
    # draw_training_and_testing()
