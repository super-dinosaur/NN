import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from torchvision import datasets, transforms
from time import time
from tqdm import tqdm

BATCH_SIZE = 512  # 批次大小
EPOCHS = 20       # 总共训练批次
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测设备

# 设置日志文件路径
log_filename = f'./ImprovedConvNet_{BATCH_SIZE}_{EPOCHS}_{DEVICE}_log.txt'

# 清空日志文件内容
open(log_filename, 'w').close()

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    filename=log_filename, 
                    filemode='a')  # 追加模式

class ImprovedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1.卷两层再池化
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # 2.卷两层再池化
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 3.卷两层再池化
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # # 全连接层
        # self.fc1 = nn.Linear(128 * 3 * 3, 256)
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x)) #128*3*3->256
        # x = self.dropout(x)
        # x = F.log_softmax(self.fc2(x), dim=1)
        return x

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