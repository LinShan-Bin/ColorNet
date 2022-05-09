import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import scipy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, model, class_num, criterion, optimizer, milestones=[10, 30], gamma=0.1, reinforcement=None):
        self.model = model
        self.class_num = class_num
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reinfocement = reinforcement
        self.writer = SummaryWriter('./logs')
        
    def model_summary(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: {}".format(num_params))
        print(self.model)
        
    def init_model(self):
        # A naive initialization.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def train(self, train_loader, epochs, save_dir='./pretrained_model/', freq=1, test_loader=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.to(self.device)
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))
            totoal_loss = 0
            totoal_correct = 0
            totoal_samples = 0
            self.model.train()
            pbar = tqdm(train_loader)
            for i, (data, mask, opt_tags, target) in enumerate(pbar):
                data, mask, opt_tags, target = data.to(self.device), mask.to(self.device), opt_tags.to(self.device), target.squeeze().long().to(self.device)
                # Data reinforcement.
                if self.reinfocement is not None:
                    data = self.reinfocement(data)
                
                output = self.model(data * mask) * opt_tags
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                totoal_loss += loss.item()
                totoal_correct += (output.max(1)[1] == target).sum().item()
                totoal_samples += target.size(0)
                pbar.set_description(f'Loss: {totoal_loss / totoal_samples:.6f}')
                self.writer.add_scalar('train/loss (step)', loss.item(), epoch * len(train_loader) + i + 1)
                self.writer.add_scalar('train/acc (step)', (output.max(1)[1] == target).sum().item() / target.size(0), epoch * len(train_loader) + i + 1)
            
            if (epoch + 1) % freq == 0:
                avg_loss = totoal_loss / len(train_loader)
                acc = totoal_correct / totoal_samples
                self.writer.add_scalar('train/loss', avg_loss, epoch + 1)
                self.writer.add_scalar('train/acc', acc, epoch + 1)
                
                save_name = save_dir + 'model_' + str(epoch + 1) + '.pth' if (epoch + 1) % 5 == 0 else save_dir + 'model_final.pth'
                torch.save(self.model.state_dict(), save_name)
                print('Epoch: {}. Train acc: {:.4f}. Training loss: {:.6f}'.format(epoch + 1, acc, avg_loss))
                print('Model saved.')

                if test_loader is not None:
                    self.test(test_loader, epoch=epoch)
                
            self.scheduler.step()
            

    def test(self, test_loader, detail=False, epoch=0, log_dir='./pretrained_model/'):
        self.model.eval()
        pred = []
        target = []
        total_loss = 0
        with torch.no_grad():
            for x, m, o, y in test_loader:
                x, m, o, y = x.to(self.device), m.to(self.device), o.to(self.device), y.to(self.device).long().squeeze(1)
                output = self.model(x) * o
                loss = self.criterion(output, y)
                total_loss += loss.item()
                pred.append(output.cpu().numpy())
                target.append(y.cpu().numpy())
        
        pred = np.concatenate(pred, axis=0)
        pred = scipy.special.softmax(pred, axis=1)
        target = np.concatenate(target, axis=0)
        
        if detail:
            auroc = metrics.roc_auc_score(target, pred, average='macro', multi_class='ovr')
            
            print(
                f"Classification Report:\n",
                f"{metrics.classification_report(target, pred.argmax(axis=1))}\n"
                )
            print('AUROC: {:.4f}'.format(auroc))
            print("Accuracy: {:.4f}".format(metrics.accuracy_score(target, pred.argmax(axis=1))))
            print("Avg Loss: {:.6f}".format(total_loss / len(test_loader) / test_loader.batch_size))
            
            cm = metrics.confusion_matrix(target, pred.argmax(axis=1))
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(self.classes))
            disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
            plt.show()
            
        else:
            acc = metrics.accuracy_score(target, pred.argmax(axis=1))
            loss = total_loss / len(test_loader)
            print("Accuracy: {:.4f}.\tAvg Loss: {:.6f}".format(acc, loss))
            self.writer.add_scalar('test/acc', acc, epoch + 1)
            self.writer.add_scalar('test/loss', loss, epoch + 1)
