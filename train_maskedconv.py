# %%
# !featurize dataset download 6878dbe0-2e3d-4065-92e0-1db6ec358071

# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, convnext_tiny, convnext_base


import utils
import models
import label_processor

# %%
DATA_PATH = '/home/featurize/data/medium/'
# DATA_PATH = './dataset/medium/'
embed = label_processor.CStdLib(single=True)
CLASS_NUM = embed.class_num

# TODO: Ablation experiment (Mask)
# TODO: Use other instance segmentation models (e.g. Swin)
dataset = utils.ColorfulClothesCLF(DATA_PATH, class_num=CLASS_NUM, embed=embed, train=True)
data_distribution = dataset.clean_and_analyse()
print(data_distribution)
small_len = int(len(dataset) * 1.0)
small, _ = random_split(dataset, [small_len, len(dataset) - small_len])
val_len = int(len(small) * 0.1)
train, val = random_split(small, [len(small) - val_len, val_len])

train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=6)
test_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=6)

# %%
model = convnext_tiny(pretrained=True)
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters: {}".format(num_params))
print(model)

# %%
model.classifier[2] = nn.Linear(768, CLASS_NUM)

# %%
loss_weight = (1. / data_distribution).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
# TODO: Ablation experiment
criterion = nn.CrossEntropyLoss(reduction='mean', weight=loss_weight)
# TODO: Add an introductory task: predict the RGB value.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
trainer = utils.Trainer(model, class_num=CLASS_NUM, criterion=criterion, optimizer=optimizer, milestones=[1, 5, 10])

# %%
trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=20)


