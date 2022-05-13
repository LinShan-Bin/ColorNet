# %%
from tqdm import tqdm
import torch
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader


import utils

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
maskrcnn = maskrcnn_resnet50_fpn(pretrained=True, progress=True)
maskrcnn = maskrcnn.eval().to(device)

# %%
train_data = utils.ColorfulClothesIMG(data_path='/home/featurize/data/medium/', train=True, org=False)
test_data = utils.ColorfulClothesIMG(data_path='/home/featurize/data/medium/', train=False, org=False)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=6)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=6)

# %%
# pbar = tqdm(train_loader)
# i = 0
# ones = torch.ones((1, 224, 224), dtype=torch.bool).cpu()
# with torch.no_grad():
#     for batch in pbar:
#         img, name = batch
#         img = img.to(device)
#         mask = maskrcnn(img)
#         for i, n in enumerate(name):
#             # 防止有图片检测不到物品
#             bool_mask = ones
#             try:
#                 for j in range(len(mask[i]['labels'])):
#                     if mask[i]['labels'][j] == 1:  # 1 是人的 label
#                         bool_mask = mask[i]['masks'][j] > 0.5
#                         break
#                     if mask[i]['scores'][j] < 0.7:  # 确信度小时不使用mask
#                         break
#             except:
#                 pass
#             save_path =  n.replace('jpg', 'pt')
#             torch.save(bool_mask.cpu(), save_path)

# # %%
# pbar = tqdm(test_loader)
# i = 0
# ones = torch.ones((1, 224, 224), dtype=torch.bool).cpu()
# with torch.no_grad():
#     for batch in pbar:
#         img, name = batch
#         img = img.to(device)
#         mask = maskrcnn(img)
#         for i, n in enumerate(name):
#             # 防止有图片检测不到物品
#             bool_mask = ones
#             try:
#                 for j in range(len(mask[i]['labels'])):
#                     if mask[i]['labels'][j] == 1:  # 1 是人的 label
#                         bool_mask = mask[i]['masks'][j] > 0.5
#                         break
#                     if mask[i]['scores'][j] < 0.7:  # 确信度小时不使用mask
#                         break
#             except:
#                 pass
#             save_path =  n.replace('jpg', 'pt')
#             torch.save(bool_mask.cpu(), save_path)

# %%
pbar = tqdm(train_loader)
i = 0
ones = torch.ones((1, 224, 224), dtype=torch.bool).cpu()
with torch.no_grad():
    for batch in pbar:
        img, name = batch
        img = img.to(device)
        mask = maskrcnn(img)
        for i, n in enumerate(name):
            # 防止有图片检测不到物品
            try:
                bool_mask = mask[i]['masks'][0] > 0.5
            except:
                bool_mask = ones
            save_path =  n.replace('.jpg', '_0.pt')
            torch.save(bool_mask.cpu(), save_path)

# %%
pbar = tqdm(test_loader)
i = 0
ones = torch.ones((1, 224, 224), dtype=torch.bool).cpu()
with torch.no_grad():
    for batch in pbar:
        img, name = batch
        img = img.to(device)
        mask = maskrcnn(img)
        for i, n in enumerate(name):
            # 防止有图片检测不到物品
            bool_mask = ones
            try:
                bool_mask = mask[i]['masks'][0] > 0.5
            except:
                bool_mask = ones
            save_path =  n.replace('.jpg', '_0.pt')
            torch.save(bool_mask.cpu(), save_path)
