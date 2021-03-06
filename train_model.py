import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.models import resnet50, convnext_tiny, convnext_base, convnext_large
from torchvision.transforms import RandomCrop, RandomRotation, RandomHorizontalFlip, Compose
from torch.utils.tensorboard import SummaryWriter


import utils
import label_processor

torch.manual_seed(42)

DATA_PATH = '/home/featurize/data/medium/'
# DATA_PATH = './dataset/medium/'
embed = label_processor.CStdLib(single=True)
CLASS_NUM = embed.class_num


def train(model, save_dir, bs, lr, ms):
    # TODO: (Ablation experiment) Mask
    # The model works better without masks.
    dataset = utils.ColorfulClothesCLF(DATA_PATH, class_num=CLASS_NUM, embed=embed, train=True, mask=False)
    data_distribution = dataset.clean_and_analyse()
    print(data_distribution)

    sample_weights = dataset.sample_weights

    weighted_sample = WeightedRandomSampler(sample_weights, num_samples=len(dataset), generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(dataset, batch_size=bs, num_workers=6, sampler=weighted_sample)

    augmentation = Compose([
        RandomCrop(224, padding=64),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=30),
    ])
    # TODO: (Ablation experiment) Sample
    # loss_weight = (1. / data_distribution).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # TODO: (optimize) Add an introductory task: predict the RGB value.
    # No need. Since the model is already well trained.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = utils.Trainer(model, save_dir=save_dir, class_num=CLASS_NUM, augmentation=augmentation, criterion=criterion, optimizer=optimizer, milestones=ms, gamma=0.3)

    trainer.train(train_loader=train_loader, test_loader=None, epochs=30)


def exp_resnet50():
    model = resnet50(pretrained=True)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_params))
    print(model)

    model.fc = nn.Linear(2048, CLASS_NUM)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    train(model, save_dir='./pretrained_model/ResNet50/', bs=64, lr=1e-4, ms=[1, 4, 7, 10])


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
    train(model, save_dir='./pretrained_model/ConvX_base/', bs=64, lr=1e-4, ms=[1, 4, 7, 10])


def exp_convx_large():
    model = convnext_base(pretrained=True)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_params))
    print(model)

    # Freeze the first 5 feature layers (7 in total) and change the classifier.
    model.classifier[2] = nn.Linear(1536, CLASS_NUM)
    for i in range(5):
        for param in model.features[i].parameters():
            param.requires_grad = False
    train(model, save_dir='./pretrained_model/ConvX_large/', bs=64, lr=1e-4, ms=[1, 4, 7, 10])
    
    
if __name__ == '__main__':
    # exp_resnet50()
    exp_convx_tiny()
    # exp_convx_base()
    # exp_convx_large()
