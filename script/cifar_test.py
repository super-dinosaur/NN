import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from time import time
from fierce_fnet import ImprovedConvNet

# 如果你有GPU可以用，就写 'cuda'，否则 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128  # CIFAR-10可用128、256视显存而定
EPOCHS = 20

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
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

        running_loss += loss.item()
        # 计算准确率
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            acc = 100. * correct / total
            print(f"Epoch [{epoch}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy

if __name__ == "__main__":
    # ========== 数据集与DataLoader ==========
    # 这里采用简单的 Normalization，如果想提升效果，可以加数据增广
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 适度的数据增广
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))  # CIFAR-10官方均值方差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data_cifar', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data_cifar', train=False,
                                    download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ========== 模型初始化 ==========
    model = ImprovedConvNet().to(DEVICE)  # 使用上面自定义的网络(已改成3通道输入)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 也可以用学习率调度器
    # from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # ========== 训练与验证循环 ==========
    time_start = time()
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, DEVICE, test_loader)

        # 如果使用scheduler，则在每个epoch结束后step
        # scheduler.step()

    time_end = time()
    print(f"Total training time: {time_end - time_start:.2f}s")