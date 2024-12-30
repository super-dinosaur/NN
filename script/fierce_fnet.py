import torch.nn as nn
import torch.nn.functional as F

class ImprovedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 卷两层再池化
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 2. 卷两层再池化
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 3. 卷两层再池化
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 3 * 3, 512)  # 4608 -> 512
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # x = F.relu(self.bn5(self.conv5(x)))
        # x = self.pool(F.relu(self.bn6(self.conv6(x))))

        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.log_softmax(self.fc2(x), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
        #283.28s