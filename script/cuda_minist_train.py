import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from time import time
from torchvision import datasets, transforms
from icecream import ic
from tqdm import tqdm

BATCH_SIZE = 512  # 批次大小
EPOCHS = 20       # 总共训练批次
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测设备
#DEVICE = 'cpu'  # 强制使用CPU

# 设置日志文件路径
log_filename = f'./ConvNet_{BATCH_SIZE}_{EPOCHS}_{DEVICE}_log.txt'

# 清空日志文件内容
open(log_filename, 'w').close()

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    filename=log_filename, 
                    filemode='a')  # 追加模式

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28x28
        self.pool1 = nn.MaxPool2d(2, 2)             # 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)            # 10x10
        self.pool2 = nn.MaxPool2d(2, 2)             # 5x5
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)   #120*1*1，两个1是图片最后被降到的尺寸
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)         #x_init.shape = (512:batch_size, 1:channel, 28:height, 28:width)
        out = self.conv1(x)        # 24
        out = F.relu(out)
        out = self.pool1(out)     # 12
        out = self.conv2(out)     # 10
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out) 
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   #optimizer.zero_grad()是用来清空上一步的残余更新参数值
        output = model(data)
        loss = F.nll_loss(output, target)   #nll_loss是负对数似然损失函数
        loss.backward()
        optimizer.step()    #optimizer.step()更新所有的参数
        if (batch_idx + 1) % 30 == 0:   #每30个batch打印一次训练结果
            info = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                     f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            logging.info(info)
            #print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  #f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()    #model.eval()是用来固定模型的,不会改变权值,不会进行dropout,不会进行batchnorm,不会进行梯度下降,只是单纯的前向传播
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    info = f'\nTest set: Average loss: {test_loss:.4f}, ' \
              f'Accuracy: {correct}/{len(test_loader.dataset)} ' \
                f'({100. * correct / len(test_loader.dataset):.0f}%)\n'
    logging.info(info)
    '''
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(test_loader.dataset)} '
            f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    '''

if __name__ == "__main__":
    time_start = time()
    print(f"Using device: {DEVICE}")  # 输出当前使用的设备
    logging.info(f"Using device: {DEVICE}")  # 输出当前使用的设备

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    model = ConvNet().to(DEVICE)  # 将模型移动到指定设备
    optimizer = optim.Adam(model.parameters())  # 使用Adam优化器,optimizer是用来更新参数的

    #for epoch in range(1, EPOCHS + 1):
    for epoch in tqdm(range(1, EPOCHS + 1)):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)

    time_end = time()
    print(f"Total time: {time_end - time_start:.2f}s")  # 计算总耗时
    logging.info(f"Total time: {time_end - time_start:.2f}s")  # 计算总耗时