import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import scipy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os


class Trainer(object):
    def __init__(self, model, class_num, criterion, optimizer, milestones=[10, 30], gamma=0.1, last_epoch=-1, reinforcement=None):
        self.model = model
        self.class_num = class_num
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reinfocement = reinforcement
        self.training_loss = []
        self.training_acc = []
        self.test_loss = []
        self.test_acc = []
        
    def model_summary(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: {}".format(num_params))
        print(self.model)
        
    def init_model(self):
        # A naive initialization.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def train(self, train_loader, epochs, save=True, save_dir='./pretrained_model/', freq=1, test_loader=None):
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
            for data, _, target in pbar:
                data, target = data.to(self.device), target.to(self.device).squeeze().long()
                # Data reinforcement.
                if self.reinfocement is not None:
                    data = self.reinfocement(data)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                totoal_loss += loss.item()
                totoal_correct += (output.max(1)[1] == target).sum().item()
                totoal_samples += target.size(0)
                
            
            if (epoch + 1) % freq == 0:
                avg_loss = totoal_loss / len(train_loader)
                acc = totoal_correct / totoal_samples
                self.training_loss.append((epoch + 1, avg_loss))
                self.training_acc.append((epoch + 1, acc))
                np.save(save_dir + 'loss.npy', np.array(self.training_loss))
                np.save(save_dir + 'acc.npy', np.array(self.training_acc))
                
                save_name = save_dir + 'model_' + str(epoch + 1) + '.pth' if (epoch + 1) % 5 == 0 else save_dir + 'model_final.pth'
                torch.save(self.model.state_dict(), save_name)
                print('Epoch: {}. Train acc: {:.4f}. Training loss: {:.6f}'.format(epoch + 1, acc, avg_loss))
                print('Model saved.')

                if test_loader is not None:
                    self.test(test_loader, epoch=epoch)
                
            self.scheduler.step()
            

    def test(self, test_loader, detail=False, epoch=0):
        self.model.eval()
        pred = []
        target = []
        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device).squeeze().long()
                output = self.model(x)
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
            self.test_acc.append((epoch + 1, acc))
            self.test_loss.append((epoch + 1, loss))
            print("Accuracy: {:.4f}.\tAvg Loss: {:.6f}".format(acc, loss))

    def log_curv(self, log=None):
        if log is None:
            log = self.training_loss
        x = [i for i, _ in log]
        y = [j for _, j in log]
        plt.plot(x, y)
        plt.title('Training loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.show()
