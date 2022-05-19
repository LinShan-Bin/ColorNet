import json
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, convnext_tiny, convnext_base

import utils
import label_processor


DATA_PATH = '/home/featurize/data/medium/'
# DATA_PATH = './dataset/medium/'
embed = label_processor.CStdLib(single=False)
CLASS_NUM = embed.class_num
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
dataset = utils.ColorfulClothesTest(DATA_PATH, class_num=CLASS_NUM, embed=embed, mask=False)
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
            # softmaxed = F.softmax(logits, dim=1) * muls
            # softmaxed = softmaxed.detach().cpu().numpy()
            # softmaxed = (logits * muls).cpu().numpy()
            logits = logits.cpu().numpy()
            # 生成标签
            for i in range(len(img_ids)):
                # pid = pro_id[i]
                # id = img_ids[i]
                # name = img_ns[i]
                # sf = softmaxed[i]
                # opt_tags = test_json[pid]['optional_tags']
                # prob = np.zeros(len(opt_tags))  # 不一定是“真概率”
                # for i, tag in enumerate(opt_tags):
                #     emb = embed(tag)
                #     if len(emb) == 0:
                #         prob[i] = 10
                #         # 这么定是希望模型做排除法，其他颜色概率不大时选择未知的标签。10 是尝试出来效果比较好的值。
                #     else:
                #         prob[i] = np.mean(sf[emb])  # 一个标签也适用
                # tag_id = np.argmax(prob)
                # test_json[pid]['imgs_tags'][id][name] = opt_tags[tag_id]
                logit_i = ''
                for j in range(CLASS_NUM):
                    logit_i = logit_i + '{:.6f} '.format(logits[i][j].item())
                pid = pro_id[i]
                id = img_ids[i]
                name = img_ns[i]
                test_json[pid]['imgs_tags'][id][name] = logit_i


    with open('./result.json', 'w+', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # For ResNet50
    # model = resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, CLASS_NUM)
    # check_point = torch.load('pretrained_model/ResNet50/checkpoint.pth')
    # state_dict = check_point['model_state_dict']
    # model.load_state_dict(state_dict)
    # get_json(model)

    # For MaskedConvX_tiny
    model = convnext_tiny(pretrained=False)
    model.classifier[2] = nn.Linear(768, CLASS_NUM)
    check_point = torch.load('pretrained_model/ConvX_tiny_AdamW/checkpoint.pth')
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)
    get_json(model)

    # For MaskedConvX_base
    # model = convnext_base(pretrained=False)
    # model.classifier[2] = nn.Linear(1024, CLASS_NUM)
    # check_point = torch.load('pretrained_model/MaskedConvX_base/checkpoint.pth')
    # state_dict = check_point['model_state_dict']
    # model.load_state_dict(state_dict)
    # get_json(model)
