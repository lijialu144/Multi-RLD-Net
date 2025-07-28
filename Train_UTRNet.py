import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import matplotlib.pyplot as plt
sys.path.append('../..')
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import crnn
from models import UTRNet_RLDNet
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd
np.random.seed(1)

def FWCLoss(pred,target, W, count_pos, count_neg):                                                                                #对预测的概率进行Sigmoid激活
    m = nn.Sigmoid()
    lossinput = m(pred)
    ratio= count_neg/(count_pos + count_neg)                                                              #负样本的比例
    W = W.unsqueeze(1).expand_as(target)                                                                   #将1维的展开为2维，相当于复制了一个维度  500变为500*2
    L = - (ratio * target * torch.log(lossinput + 1e-10) * torch.pow(input=(1-lossinput),exponent=2) +
           (1 - ratio) * (1 - target) * torch.log(1 - lossinput + 1e-10) * torch.pow(input=lossinput,exponent=2))
    output = torch.mean(L * W)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=15, help='epoch to run[default: 50]')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate[default: 3e-4]')
parser.add_argument('--save_path', default='best_models/UTRNet', help='model param path')
parser.add_argument('--data_path', default='data/UTRNet/', help='dataset path')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')
parser.add_argument('--adjust_lr', type=bool, default=True, help='adjust learning rate')
parser.add_argument('--learning_rate_steps', type=list, default=[10,15,20,25,30,35,40,45], help='learning_rate_steps')
parser.add_argument('--learning_rate_gamma', type=float, default=0.5, help='learning_rate_gamma ')
flags = parser.parse_args()

class Trainer(object):
    def __init__(self, flags):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = flags.learning_rate
        print('======distance rnn========')
        self.model = UTRNet_RLDNet.ConvDisRNN(batch_size=flags.batch_size, in_channels=6, out_channels=64,
                                                 kernel_size=3, num_classes=2, time_len=10).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def kappa(confusion_matrix):
        pe_rows = np.sum(confusion_matrix, axis=0)
        pe_cols = np.sum(confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):     #optim=adam,epoch=epoch, steps=[10,15,20,25,30,35,40,45], lr_gamma=0.5
        if epoch == 0:
            self.lr = 0.001
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:    #optimizer.param_groups包括['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize','foreach','capturable']
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def train(self):
        print('Training..................')
        path_train = flags.data_path
        print('path_train', path_train)
        with open(os.path.join(path_train, 'all_sample.pickle'), 'rb') as file:
            all_sample = pickle.load(file)
        with open(os.path.join(path_train, 'all_label.pickle'), 'rb') as file:
            all_label = pickle.load(file) # 0 is unchanged, 1 is changed
        with open(os.path.join(path_train, 'all_W.pickle'), 'rb') as file:
            all_W = pickle.load(file)

        all_sample = torch.from_numpy(all_sample)
        all_sample = all_sample.permute([0, 1, 4, 2, 3])
        all_sz = all_label.shape[0]
        print('all_sample', all_sample.shape)

        all_idx = np.arange(0, all_sz)
        np.random.seed(1)
        np.random.shuffle(all_idx)
        train_idx = all_idx[0:int(0.8 * all_sz)]
        val_idx = all_idx[int(0.8 * all_sz):all_sz]
        train_sample = all_sample[:, train_idx, :, :, :]
        val_sample = all_sample[:, val_idx, :, :, :]
        train_label = all_label[train_idx]
        val_label = all_label[val_idx]
        train_W = all_W[train_idx]  # 训练W（58875*1）
        val_W = all_W[val_idx]  # 验证W（21444*1）

        best_loss = 2000
        best_f1 = 0
        BATCH_SZ = flags.batch_size
        iter_train_epoch = train_label.size // BATCH_SZ
        iter_val_epoch = val_label.size // BATCH_SZ

        model_root = osp.join(flags.save_path)
        train_Loss_list = []
        val_Loss_list = []
        val_Accuracy_list = []

        for epoch in tqdm(range(0, flags.max_epoch)):
            print('self.lr', self.lr)
            train_ave_loss = 0
            train_ave_accuracy = 0
            train_ave_f1 = 0
            val_ave_loss = 0
            val_ave_accuracy = 0
            val_ave_f1 = 0
            if flags.adjust_lr:
                self.adjust_learning_rate(self.optim, epoch, flags.learning_rate_steps, flags.learning_rate_gamma)
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

            train_idx = np.arange(0, train_idx.size)
            np.random.seed(1)
            np.random.shuffle(train_idx)
            val_idx = np.arange(0, val_idx.size)
            np.random.seed(1)
            np.random.shuffle(val_idx)

            for _iter in range(iter_train_epoch):
                start_idx = _iter * BATCH_SZ                                                                                 #一个batch开始训练的下标
                end_idx = (_iter + 1) * BATCH_SZ                                                                             #一个batch结束训练的下标
                batch_train = train_sample[:, train_idx[start_idx:end_idx], :, :, :]                                         #一个batch内的训练样本
                batch_label = train_label[train_idx[start_idx:end_idx]]                                                      #一个batch内的训练标签
                batch_W = train_W[train_idx[start_idx:end_idx]]

                batch_train = torch.as_tensor(batch_train, dtype=torch.float32).to(self.device)                               #将训练样本转为cuda
                batch_label = torch.as_tensor(batch_label, dtype=torch.long).to(self.device)                                  #将训练标签转为cuda
                batch_W = torch.as_tensor(batch_W, dtype=torch.float32).to(self.device)

                train_epoch_loss, train_accuracy, train_f1 = self.train_epoch(batch_train, batch_label, batch_W)   #训练一个batch并计算一个batch的loss、oa、f1

                train_ave_loss += train_epoch_loss
                train_ave_accuracy += train_accuracy
                train_ave_f1 += train_f1

            for _iter in range(iter_val_epoch):
                start_idx = _iter * BATCH_SZ
                end_idx = (_iter + 1) * BATCH_SZ
                batch_val = val_sample[:, val_idx[start_idx:end_idx], :, :, :]
                batch_label = val_label[val_idx[start_idx:end_idx]]
                batch_W = val_W[val_idx[start_idx:end_idx]]

                batch_val = torch.as_tensor(batch_val, dtype=torch.float32).to(self.device)
                batch_label = torch.as_tensor(batch_label, dtype=torch.long).to(self.device)
                batch_W = torch.as_tensor(batch_W, dtype=torch.float32).to(self.device)

                val_epoch_loss, val_accuracy, val_f1 = self.val_epoch(batch_val, batch_label, batch_W)
                val_ave_loss += val_epoch_loss
                val_ave_accuracy += val_accuracy
                val_ave_f1 += val_f1

            best_model1 = '%s/best_model_01.pth' % (model_root)
            best_model2 = '%s/best_model_02.pth' % (model_root)
            best_model_last2 = '%s/last_model_02.pth' % (model_root)
            best_model_last1 = '%s/last_model_01.pth' % (model_root)

            train_ave_loss /= iter_train_epoch
            train_ave_accuracy /= iter_train_epoch
            train_ave_f1 /= iter_train_epoch

            val_ave_loss /= iter_val_epoch
            val_ave_accuracy /= iter_val_epoch
            val_ave_f1 /= iter_val_epoch

            train_Loss_list.append(train_ave_loss)
            val_Loss_list.append(val_ave_loss)
            val_Accuracy_list.append(val_ave_accuracy)
            print("train_ave_loss：%.4f    train_ave_accuracy：%.4f  train_ave_f1：%.4f  " % (
                train_ave_loss, train_ave_accuracy, train_ave_f1))
            print("val_ave_loss：%.4f    val_ave_accuracy：%.4f   val_ave_f1：%.4f" % (
                val_ave_loss, val_ave_accuracy, val_ave_f1))

            if val_ave_loss < best_loss:
                best_loss = val_ave_loss
                torch.save(self.model.state_dict(), best_model1)
                print("best loss model is saved")
            if val_ave_f1 > best_f1:
                best_f1 = val_ave_f1
                torch.save(self.model.state_dict(), best_model2)
                print("best f1 model is saved")
            if (flags.max_epoch-epoch)==2:
                torch.save(self.model.state_dict(), best_model_last2)
                print("last second model is saved")
            if (flags.max_epoch-epoch)==1:
                torch.save(self.model.state_dict(), best_model_last1)
                print("last first model is saved")
        plt.title('train loss & val loss')
        plt.plot(train_Loss_list)
        plt.plot(val_Loss_list)
        plt.savefig('best_models/UTRNet/loss.png')
        plt.show()

    def train_epoch(self, batch_train,batch_label, batch_W):
        self.model.train()
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        self.optim.zero_grad()
        pred1, pred2, out = self.model(batch_train)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        target = torch.eye(2)[batch_label, :].to(self.device)

        loss1 = FWCLoss(pred1, target, batch_W, count_pos, count_neg)
        loss2 = FWCLoss(pred2, target, batch_W, count_pos, count_neg)
        loss3 = FWCLoss(out, target, batch_W, count_pos, count_neg)
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
        self.optim.step()

        pos_list += out[:, 1].data.cpu().numpy().tolist()
        pred_list += out.data.max(1)[1].data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        train_epoch_loss = loss.item()
        accuracy = accuracy_score(target_list, pred_list)
        f1 = f1_score(target_list, pred_list, pos_label=1)
        return train_epoch_loss, accuracy, f1

    def val_epoch(self, batch_val, batch_label, batch_W):
        self.model.eval()
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        self.optim.zero_grad()

        pred1, pred2, out = self.model(batch_val)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        target = torch.eye(2)[batch_label, :].to(self.device)

        loss1 = FWCLoss(pred1, target, batch_W, count_pos, count_neg)
        loss2 = FWCLoss(pred2, target, batch_W, count_pos, count_neg)
        loss3 = FWCLoss(out, target, batch_W, count_pos, count_neg)
        loss = loss1 + loss2 + loss3

        pred_cls = out.data.max(1)[1]
        pos_list += out[:, 1].data.cpu().numpy().tolist()
        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        val_epoch_loss = loss.item()
        OA = accuracy_score(target_list, pred_list)
        f1 = f1_score(target_list, pred_list, pos_label=1)
        return val_epoch_loss, OA, f1


if __name__ == '__main__':
    # seed_everything(seed=1, cuda=True)
    trainer = Trainer(flags)
    trainer.train()
