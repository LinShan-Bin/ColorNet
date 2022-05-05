import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained = True)
        self.net = nn.Sequential(*list(resnet50.children())[:-2])
        for param in self.net.parameters():
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transfrom = nn.Linear(2048, 300)
        
    def forward(self, x):
        x = self.net(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.transfrom(x)
        return x


if __name__ == '__main__':
    model = ResNet50()
    print(model)
    test_tensor = torch.randn(1, 3, 224, 224)
    res = model(test_tensor)
    print(res.shape)
