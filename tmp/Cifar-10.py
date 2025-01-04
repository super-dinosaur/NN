import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import logging

# 设置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

BATCH_SIZE = 512  # 批次大小
EPOCHS = 20  # 总共训练批次
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有GPU，则使用GPU，否则使用CPU

# 定义一个通用的激活函数接口
def activation_fn(activation_type):
    if activation_type == 'ReLU':
        return F.relu
    elif activation_type == 'GELU':
        return F.gelu
    elif activation_type == 'Tanh':
        return torch.tanh
    elif activation_type == 'ELU':
        return F.elu
    else:
        raise ValueError("Unsupported activation function")

class ConvNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # CIFAR-10有3个通道
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.activation_fn = activation_fn

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.activation_fn(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.activation_fn(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.activation_fn(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.activation_fn(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

def train(model, device, train_loader, optimizer, epoch, activation_name, train_losses):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx + 1) % 30 == 0:
            log_message = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}, Activation: {activation_name}'
            print(log_message)
            logging.info(log_message)
    train_losses.append(epoch_loss / len(train_loader))

def test(model, device, test_loader, activation_name, test_losses, test_accuracies):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)
    log_message = f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%), Activation: {activation_name}\n'
    print(log_message)
    logging.info(log_message)

if __name__ == "__main__":
    # activation_types = ['ReLU', 'GELU', 'Tanh', 'ELU']
    activation_types = ['ELU']
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=BATCH_SIZE, shuffle=False)

    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    for activation_type in activation_types:
        model = ConvNet(activation_fn(activation_type)).to(DEVICE)
        optimizer = optim.Adam(model.parameters())
        train_losses = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(1, EPOCHS + 1):
            train(model, DEVICE, train_loader, optimizer, epoch, activation_type, train_losses)
            test(model, DEVICE, test_loader, activation_type, test_losses, test_accuracies)
        
        # 为每个激活函数绘制训练损失图
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, EPOCHS + 1), train_losses, label=f'Train Loss - {activation_type}')
        plt.title(f'Training Loss - {activation_type}')
        plt.xlabel('Epoch')
        plt.xlim(1, EPOCHS + 1)
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{activation_type}_train_loss.png'))
        plt.close()

        # 为每个激活函数绘制测试损失和准确率图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, EPOCHS + 1), test_losses, label=f'Test Loss - {activation_type}')
        plt.title(f'Testing Loss - {activation_type}')
        plt.xlabel('Epoch')
        plt.xlim(1, EPOCHS + 1)
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, EPOCHS + 1), test_accuracies, label=f'Test Accuracy - {activation_type}', color='orange')
        plt.title(f'Testing Accuracy - {activation_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{activation_type}_test_loss_accuracy.png'))
        plt.close()