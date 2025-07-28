import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import matplotlib.pyplot as plt
sys.path.append('../..')
import yaml
import pickle
import torch
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score
import crnn
from models import UTRNet_RLDNet
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=50, help='epoch to run[default: 50]')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate[default: 3e-4]')
parser.add_argument('--save_path', default='best_model', help='model param path')
parser.add_argument('--data_path', default='data/UTRNet/', help='dataset path')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')
parser.add_argument('--adjust_lr', type=bool, default=True, help='adjust learning rate')
parser.add_argument('--learning_rate_steps', type=list, default=[10,15,20,25,30,35,40,45], help='learning_rate_steps')
parser.add_argument('--learning_rate_gamma', type=float, default=0.5, help='learning_rate_gamma ')
flags = parser.parse_args()
class Trainer(object):
    def __init__(self, cfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.001
        self.model = UTRNet_RLDNet.ConvDisRNN(batch_size=flags.batch_size, in_channels=6, out_channels=64, kernel_size=3, num_classes=2, time_len=10).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def kappa(confusion_matrix):
        pe_rows = np.sum(confusion_matrix, axis=0)
        pe_cols = np.sum(confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch == 0:
            self.lr = 0.001
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def change_map(self):

        print('Testing..................')

        model_root = osp.join('best_models')
        best_model_pth = '%s/UTRNet_RLD_Net.pth' % (model_root)
        self.model.load_state_dict(torch.load(best_model_pth))

        with open(os.path.join(flags.data_path + '/all_sample/Scene9', 'all_map.pickle'), 'rb') as file:
            all_sample = pickle.load(file)
            all_sample = all_sample[:, :, :, :, 0:6]
            print('all_sample', all_sample.shape)
        all_sz = all_sample.shape[1]
        allmap_1 = loadmat(flags.data_path + '/all_sample/Scene9' + '/' + 'data_9.mat')['allmap']
        allmap_1 = allmap_1[0][0][0]
        print('allmap_1', allmap_1.shape)
        BATCH_SZ = allmap_1.shape[0]
        iter_in_test = all_sz // BATCH_SZ
        test_idx = np.arange(0, all_sz)
        test_sample = torch.from_numpy(all_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])
        pred = []
        for _iter in range(iter_in_test):
            start_idx = _iter * BATCH_SZ
            end_idx = (_iter + 1) * BATCH_SZ
            batch_val = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_val = torch.as_tensor(batch_val, dtype=torch.float32)
            batch_val = batch_val.to(self.device)

            pred_list = self.change_map_epoch(batch_val)
            pred.extend(pred_list)

        pred_label = np.array(pred)
        all_map = np.reshape(pred_label, (allmap_1.shape[0], allmap_1.shape[1]))
        all_map = np.asarray(all_map)
        io.imsave('UTRNet_OLDNet_9.tif', all_map)
        #
        # height, width = all_map.shape
        # plt.imshow(all_map, aspect='equal', cmap='gray')
        # plt.axis('off')
        # plt.gcf().set_size_inches(width / 100, height / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('all_map.jpg', transparent=True, dpi=600, pad_inches=0)
        plt.imshow(all_map)
        plt.show()

    def change_map_epoch(self, batch_train):
        self.model.eval()
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        self.optim.zero_grad()

        pred1, pred2, out = self.model(batch_train)
        pos_list += out[:, 1].data.cpu().numpy().tolist()
        pred_list += out.data.max(1)[1].data.cpu().numpy().tolist()

        return pred_list

    def test(self):
        print('Testing..................')
        model_root = osp.join(flags.save_path)
        best_model_pth = '%s/UTRNet_RLD_Net.pth' % (model_root)
        print('best_model_pth',best_model_pth)
        self.model.load_state_dict(torch.load(best_model_pth))

        path_test = flags.data_path
        print('path_test',path_test)
        with open(os.path.join(path_test, 'test_sample.pickle'), 'rb') as file:
            test_sample = pickle.load(file)
            test_sample = test_sample[:, :, :, :, 0:6]
        with open(os.path.join(path_test, 'test_label.pickle'), 'rb') as file:
            test_label = pickle.load(file)

        test_sz = test_label.shape[0]
        BATCH_SZ = flags.batch_size
        iter_in_test = test_sz // BATCH_SZ
        test_idx = np.arange(0, test_sz)

        test_sample = torch.from_numpy(test_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])
        test_label = np.reshape(test_label, (-1))

        pred = []
        real = []

        for _iter in range(iter_in_test):
            start_idx = _iter * BATCH_SZ
            end_idx = (_iter + 1) * BATCH_SZ
            batch_val = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_label = test_label[test_idx[start_idx:end_idx]]
            batch_val = torch.as_tensor(batch_val, dtype=torch.float32).to(self.device)
            batch_label = torch.as_tensor(batch_label, dtype=torch.int64).to(self.device)

            pred_list, target_list = self.test_epoch(batch_val, batch_label)
            pred.extend(pred_list)
            real.extend(target_list)
        print('pred', len(pred))
        print('real', len(real))
        OA = accuracy_score(real, pred)
        precision = precision_score(real, pred, pos_label=1)
        recall = recall_score(real, pred, pos_label=1)
        f1 = f1_score(real, pred, pos_label=1)

        Kappa = Trainer.kappa(confusion_matrix(real, pred))
        print(confusion_matrix(real, pred))
        print('precision: %.4f   recall: %.4f  OA: %.4f   F1: %.4f  Kappa: %.4f  ' % (precision, recall, OA, f1, Kappa))

    def test_epoch(self, batch_train, batch_label):
        self.model.eval()
        pred_list, target_list, loss_list, pos_list, attn_list, attn1_list = [], [], [], [], [], []
        self.optim.zero_grad()

        pred1, pred2, out = self.model(batch_train)
        pred_list = out.data.max(1)[1].data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        return pred_list, target_list

if __name__ == '__main__':
    trainer = Trainer(flags)
    trainer.test()
    # trainer.change_map()  # get the change map
