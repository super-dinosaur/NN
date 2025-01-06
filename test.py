import os
import os.path as osp
import torch
import local_setting
import torch.nn as nn
import inspect
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from icecream import ic

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),#来自官网参数
#     "val": transforms.Compose([transforms.Resize(256),#将最小边长缩放到256
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# image_path = osp.join(local_setting.PATH_PROJECT,'LOL','our485','low')
# ic(image_path)

# train_dataset = datasets.ImageFolder(root=osp.join(image_path, ".."),
#                                      transform=data_transform["train"])
# ic(train_dataset.find_classes(osp.join(image_path, "..")))
# #ic| train_dataset.find_classes(osp.join(image_path, "..")): (['high', 'low'], {'high': 0, 'low': 1})


# ------------------------------------------------------------------------------------------------------------------------------

# # With Learnable Parameters
# m = nn.BatchNorm2d(100)
# # Without Learnable Parameters
# m = nn.BatchNorm2d(100, affine=False)
# input = torch.randn(20, 100, 35, 45)
# output = m(input)

# ------------------------------------------------------------------------------------------------------------------------------

import inspect
from torchvision.models import resnet
ic(inspect.getfile(resnet))
