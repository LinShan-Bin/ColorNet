import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


NUM_CLASS = 59
class ConvXEncoder(nn.Module):
    def __init__(self, checkpoint, num_class) -> None:
        super().__init__()
        model = convnext_tiny(pretrained=False)
        model.classifier[2] = nn.Linear(768, num_class)
        for param in model.parameters():
            param.requires_grad = False
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        self.model = model
    
    def forward(self, pic):
        self.model.eval()
        with torch.no_grad():
            x = self.model.features(pic)
            x = self.model.avgpool(x)
            x = self.model.classifier[0](x)  # Layer Norm
            x = self.model.classifier[1](x)  # Flatten
        return x


class ColorNet(nn.Module):
    def __init__(self, checkpoint, num_class, embed_dim, clf):
        super().__init__()
        self.pic_encoder = ConvXEncoder(checkpoint, num_class)
        self.feature_embedding = nn.Linear(768, embed_dim)
        self.clf = clf

    def forward(self, data):
        image, tag = data
        pic_embedding = self.pic_encoder(image)
        pic_embedding = self.feature_embedding(pic_embedding)
        pic_embedding = pic_embedding.unsqueeze(1)
        clf_input = torch.cat([pic_embedding, tag], dim=1)
        logits = self.clf(clf_input)
        return logits
