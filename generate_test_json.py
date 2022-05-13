import json
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, convnext_tiny, convnext_base

import utils
import models
import label_processor


DATA_PATH = '/home/featurize/data/medium/'
# DATA_PATH = './dataset/medium/'
embed = label_processor.CStdLib(single=False)
CLASS_NUM = embed.class_num
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
dataset = utils.ColorfulClothesTest(DATA_PATH, class_num=CLASS_NUM, embed=embed)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

JSON_PATH = DATA_PATH + 'test_all.json'
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    test_json = json.load(f)

def get_json(model):
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for imgs, masks, muls, pro_id, img_ids, img_ns in pbar:
            imgs, masks, muls = imgs.to(DEVICE), masks.to(DEVICE), muls.to(DEVICE)
            logits = model(imgs * masks)
            softmaxed = F.softmax(logits, dim=1) * muls
            softmaxed = softmaxed.detach().cpu().numpy()
            
            # 生成标签
            for i in range(len(img_ids)):
                pid = pro_id[i]
                id = img_ids[i]
                name = img_ns[i]
                sf = softmaxed[i]
                opt_tags = test_json[pid]['optional_tags']
                prob = np.zeros(len(opt_tags))
                for i, tag in enumerate(opt_tags):
                    emb = embed(tag)
                    if len(emb) == 0:
                        prob[i] = 1. / len(opt_tags) + 0.2  # 这么定是希望模型做排除法（似乎挺有用的，和换mask一起提升了2%）
                    else:
                        prob[i] = np.mean(sf[emb])  # 一个标签也适用
                tag_id = np.argmax(prob)
                test_json[pid]['imgs_tags'][id][name] = opt_tags[tag_id]

    with open('./result.json', 'w+', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # For MaskedConvX_tiny
    # model = convnext_tiny(pretrained=False)
    # model.classifier[2] = nn.Linear(768, CLASS_NUM)
    # check_point = torch.load('pretrained_model/MaskedConvX_tiny/checkpoint.pth')
    # state_dict = check_point['model_state_dict']
    # model.load_state_dict(state_dict)
    # get_json(model)

    # For MaskedConvX_base
    model = convnext_base(pretrained=False)
    model.classifier[2] = nn.Linear(1024, CLASS_NUM)
    check_point = torch.load('pretrained_model/MaskedConvX_base/checkpoint.pth')
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)
    get_json(model)