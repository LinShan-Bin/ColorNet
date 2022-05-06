import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchvision.models.detection import maskrcnn_resnet50_fpn


class MaskColorNet(nn.Module):
    def __init__(self, clf, mask_thres=0.5, ins_seg=maskrcnn_resnet50_fpn(pretrained=True, progress=False)):
        super().__init__()
        for param in ins_seg.parameters():
            param.requires_grad = False
        self.mask_net = ins_seg
        self.mask_thres = mask_thres
        self.clf = clf
        
    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.tensor): [batch_size, 3, height, width], Values are in range [0, 1].
        """
        self.mask_net.eval()
        mask = self.mask_net(x)
        mask = [m['masks'][0] for m in mask]
        mask = torch.stack(mask, dim=0)
        mask = mask > self.mask_thres
        mask = mask.requires_grad_(False)
        x = x * mask
        logits = self.clf(x)
        return logits


class MaxPoolingCLF(nn.Module):
    def __init__(self, num_class=16):
        super().__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=3, padding=3, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, stride=3, padding=3, bias=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.finalmax = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(64, self.num_class)
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.maxpool4(F.relu(self.conv4(x)))
        x = self.maxpool5(F.relu(self.conv5(x)))
        x = self.finalmax(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
