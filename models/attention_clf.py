import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class AttentionCLF(nn.Module):
    def __init__(self, hiddens, **kwargs) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(**kwargs)
        self.fcs = nn.Sequential()
        for i in range(len(hiddens) - 1):
            self.fcs.add_module(f'fc{i}', nn.Linear(hiddens[i], hiddens[i + 1]))
            self.fcs.add_module(f'bn{i}', nn.BatchNorm1d(hiddens[i + 1]))
            self.fcs.add_module(f'relu{i}', nn.ReLU())
        self.fcs.add_module(f'fc{len(hiddens) - 1}', nn.Linear(hiddens[-1], 1))
        self.embedding = nn.Parameter(torch.randn(1, hiddens[0]))
        
    def forward(self, x):
        emb_batch = self.embedding.repeat(x.size(0), 1).view(x.size(0), 1, -1)
        x = torch.cat([x, emb_batch], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.fcs(x)
        return x
