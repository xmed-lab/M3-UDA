import torch
import torch.nn as nn
from utils.config import opt


class Standard_classifier(nn.Module):
    def __init__(self, num_classes):
        super(Standard_classifier, self).__init__()
        # 定义分类器的层
        self.fc1 = nn.Linear(12288, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 分类器的前向传播
        x = x.view(x.size(0), -1)  # 将特征展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x