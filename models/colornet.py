from cProfile import label
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class ColorNet(pl.LightningModule):
    def __init__(self, pic_encoder, clf):
        super().__init__()
        self.pic_encoder = pic_encoder
        self.clf = clf

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        image, tag = data
        pic_embedding = self.pic_encoder(image)
        pic_embedding = pic_embedding.unsqueeze(1)
        clf_input = torch.cat([pic_embedding, tag], dim=1)
        logits = self.clf(clf_input)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, tags, labels = batch
        logits = self.forward((images, tags))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        acc = torch.mean(((logits > 0) == labels).float())
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, tags, labels = batch
        logits = self.forward((images, tags))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        acc = torch.mean(((logits > 0) == labels).float())
        self.log('val_loss', loss, on_step=True)
        self.log('val_acc', acc, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    