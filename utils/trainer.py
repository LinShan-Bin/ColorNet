import json
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import scipy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, model, class_num, criterion, optimizer, milestones=[10, 30], gamma=0.1, reinforcement=None, save_dir='./pretrained_model/'):
        self.model = model
        self.class_num = class_num
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reinfocement = reinforcement
        self.writer = SummaryWriter(save_dir + 'logs')
        self.save_dir = save_dir
        
    def model_summary(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: {}".format(num_params))
        print(self.model)
        
    def init_model(self):
        # A naive initialization.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def train(self, train_loader, epochs, freq=1, test_loader=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.model.to(self.device)
        for epoch in range(1 , epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            totoal_loss = 0
            totoal_correct = 0
            totoal_samples = 0
            self.model.train()
            pbar = tqdm(train_loader)
            for i, (data, mask, opt_tags, target, pid) in enumerate(pbar):
                data, mask, opt_tags, target = data.to(self.device), mask.to(self.device), opt_tags.to(self.device), target.squeeze().long().to(self.device)
                masked = data * mask
                # Data reinforcement.
                if self.reinfocement is not None:
                    masked = self.reinfocement(masked)
                
                output = self.model(masked) * opt_tags
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                loss_value = loss.item()
                len_batch = target.size(0)
                totoal_loss += loss_value
                totoal_samples += len_batch

                total_step = (epoch - 1) * len(train_loader) + i + 1
                pred_label = output.max(1)[1]  # max() returns a tuple (value, index)
                correct = pred_label.eq(target).sum().item()
                totoal_correct += correct
                pbar.set_description(f'Loss: {loss_value:.6f}')

                self.writer.add_scalar('train/loss (step)', loss_value, total_step)
                self.writer.add_scalar('train/acc (step)', correct / len_batch, total_step)
                vis_num = min(len_batch, 16)
                if i % 400 == 0:
                    # For visualization.
                    imgs = []  # img 和 mask 放一起
                    result_json = dict()
                    for j in range(vis_num):
                        im = data[j].detach().cpu()
                        imgs.append(im)
                        ma = mask[j].repeat(3, 1, 1).cpu()
                        imgs.append(ma)
                
                        pd = pred_label[j].item()
                        td = target[j].item()
                        result_json[pid[j]] = {'pred': pd, 'target': td}
        
                    grid = torchvision.utils.make_grid(imgs, normalize=True, scale_each=True)
                    self.writer.add_image('test/img', grid, total_step)
                    with open(self.save_dir + 'E{}S{}_pred_exp.json'.format(epoch, total_step), 'w') as f:
                        json.dump(result_json, f, indent=4)

            if epoch % freq == 0:
                avg_loss = totoal_loss / len(train_loader)
                acc = totoal_correct / totoal_samples
                self.writer.add_scalar('train/loss', avg_loss, epoch)
                self.writer.add_scalar('train/acc', acc, epoch)
                
                if epoch % 5 == 0:
                    save_name = self.save_dir + 'model_' + str(epoch) + '.pth'
                    torch.save(self.model.state_dict(), save_name)
                else:
                    check_point = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': avg_loss,
                        'acc': acc
                    }
                    torch.save(check_point, self.save_dir + 'checkpoint.pth')
                
                print('Epoch: {}. Train acc: {:.4f}. Training loss: {:.6f}'.format(epoch, acc, avg_loss))
                print('Model saved.')

                if test_loader is not None:
                    self.test(test_loader, epoch=epoch)
                
            self.scheduler.step()
            

    def test(self, test_loader, detail=False, epoch=0):
        self.model.eval()
        pred = []
        target = []
        total_loss = 0
        # For visualization.
        imgs = []  # img 和 mask 放一起
        result_json = dict()
        with torch.no_grad():
            for x, m, o, y, pid in test_loader:
                x, m, o, y = x.to(self.device), m.to(self.device), o.to(self.device), y.to(self.device).long().squeeze(1)
                output = self.model(x) * o
                loss = self.criterion(output, y)
                total_loss += loss.item()
                pred.append(output.cpu().numpy())
                target.append(y.cpu().numpy())
                
                # For visualization.
                imgs.append(x[0].cpu())
                mask = m[0].repeat(3, 1, 1).cpu()
                imgs.append(mask)
                
                pd = torch.argmax(output[0], dim=0).item()
                td = y[0].item()
                result_json[pid[0]] = {'pred': pd, 'target': td}
        
        grid = torchvision.utils.make_grid(imgs[:16], normalize=True, scale_each=True)
        self.writer.add_image('test/img', grid, epoch)
        with open(self.save_dir + 'E{}_validate_example.json'.format(epoch), 'w') as f:
            json.dump(result_json, f, indent=4)
        
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
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(self.class_num))
            disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
            plt.show()
            
        else:
            acc = metrics.accuracy_score(target, pred.argmax(axis=1))
            loss = total_loss / len(test_loader)
            print("Accuracy: {:.4f}.\tAvg Loss: {:.6f}".format(acc, loss))
            self.writer.add_scalar('test/acc', acc, epoch)
            self.writer.add_scalar('test/loss', loss, epoch)
