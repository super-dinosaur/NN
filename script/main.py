import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import json

from torchvision import datasets, transforms
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from improved_fnet import ImprovedConvNet

BATCH_SIZE = 512  # 批次大小
EPOCHS = 20       # 总共训练批次
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测设备
#DEVICE = 'cpu'  # 强制使用CPU
DESCRIPTION = "FNet0_cpu"

log_filename = f'./ImprovedConvNet_{BATCH_SIZE}_{EPOCHS}_{DESCRIPTION}_log.txt'
log_filename = f'Fnet0_cpu_log.txt'
open(log_filename, 'w').close()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    filename=log_filename, 
                    filemode='a')  # 追加模式

list_accuracy = []

def train(model, device, train_loader, optimizer, epoch, log_interval=30):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()
        
        # 计算正确预测的数量
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % log_interval == 0:
            current_loss = running_loss / log_interval
            current_acc = 100. * correct / total
            log_message = (f'Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {current_loss:.6f}\tAccuracy: {current_acc:.2f}%')
            logging.info(log_message)
            #print(log_message)
            running_loss = 0.0
            correct = 0
            total = 0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    log_message = (f'\nTest set: Average loss: {test_loss:.4f}, '
                   f'Accuracy: {correct}/{len(test_loader.dataset)} '
                   f'({test_accuracy:.2f}%)\n')
    logging.info(log_message)
    list_accuracy.append(test_accuracy)
    
    #print(log_message)
    return test_loss, test_accuracy

if __name__ == "__main__":
    # 设备配置
    print(f"Using device: {DEVICE}")
    logging.info(f"Using device: {DEVICE}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    model = ImprovedConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练与测试循环
    time_start = time()
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, DEVICE, test_loader)
        scheduler.step()
    time_end = time()
    total_time = time_end - time_start
    logging.info(f"Total training time: {total_time:.2f}s")
    print(f"Total training time: {total_time:.2f}s")
    print(log_filename)

    list_idx = [i for i in range(1, EPOCHS + 1)]
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(list_idx, list_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of FNet')
    plt.grid()
    ax = plt.gca()
    plt.text(
        0.95, 0.9, 
        f'Time spent: {total_time:.2f}s', 
        transform=ax.transAxes, 
        ha='right', 
        va='top',
        fontsize=12,    # 字体大小可自行调整
    )    
    path_save = f'./ImprovedConvNet_{BATCH_SIZE}_{EPOCHS}_{DEVICE}_{DESCRIPTION}_accuracy.png'
    plt.savefig(path_save)
    print(path_save)

    #save the list of accuracy
    with open(f'./accuracy_fnet_{BATCH_SIZE}_{EPOCHS}_{DESCRIPTION}.json', 'w') as f:
        json.dump(list_accuracy, f)


