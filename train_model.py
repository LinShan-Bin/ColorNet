# 这是一条在第三方服务器上下载数据集的命令
# !featurize dataset download 6878dbe0-2e3d-4065-92e0-1db6ec358071

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.models import convnext_tiny, convnext_base
from torchvision.transforms import RandomCrop, RandomRotation, RandomHorizontalFlip, Compose
from torch.utils.tensorboard import SummaryWriter


import utils
import models
import label_processor

torch.manual_seed(42)

DATA_PATH = '/home/featurize/data/medium/'
# DATA_PATH = './dataset/medium/'
embed = label_processor.CStdLib(single=True)
CLASS_NUM = embed.class_num


def train(model, save_dir, bs, lr, ms):
    # TODO: Ablation experiment (Mask)
    # TODO: Use other instance segmentation models (e.g. Swin)
    dataset = utils.ColorfulClothesCLF(DATA_PATH, class_num=CLASS_NUM, embed=embed, train=True, mask=False)
    data_distribution = dataset.clean_and_analyse()
    print(data_distribution)

    sample_weights = dataset.sample_weights

    weighted_sample = WeightedRandomSampler(sample_weights, num_samples=len(dataset), generator=torch.Generator().manual_seed(42))
    # 租的 A6000 跑的。钱包：┭┮﹏┭┮
    train_loader = DataLoader(dataset, batch_size=bs, num_workers=6, sampler=weighted_sample)

    reinforce = Compose([
        RandomCrop(224, padding=64),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=30),
    ])
    # TODO: Ablation experiment
    # loss_weight = (1. / data_distribution).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # TODO: Add an introductory task: predict the RGB value.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = utils.Trainer(model, save_dir=save_dir, class_num=CLASS_NUM, reinforcement=reinforce, criterion=criterion, optimizer=optimizer, milestones=ms, gamma=0.3)

    trainer.train(train_loader=train_loader, test_loader=None, epochs=30)

def exp_convx_tiny():
    model = convnext_tiny(pretrained=True)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_params))
    print(model)

    # Freeze the first 5 feature layers (7 in total) and change the classifier.
    model.classifier[2] = nn.Linear(768, CLASS_NUM)
    for i in range(5):
        for param in model.features[i].parameters():
            param.requires_grad = False
    train(model, save_dir='./pretrained_model/ConvX_tiny/', bs=64, lr=1e-4, ms=[1, 4, 7, 10])

def exp_convx_base():
    model = convnext_base(pretrained=True)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_params))
    print(model)

    # Freeze the first 5 feature layers (7 in total) and change the classifier.
    model.classifier[2] = nn.Linear(1024, CLASS_NUM)
    for i in range(5):
        for param in model.features[i].parameters():
            param.requires_grad = False
    train(model, save_dir='./pretrained_model/MaskedConvX_base/', bs=128, lr=3e-5, ms=[1, 3, 5, 7, 10])


if __name__ == '__main__':
    exp_convx_tiny()
    # exp_convx_base()
