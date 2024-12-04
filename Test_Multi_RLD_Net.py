import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import matplotlib.pyplot as plt
sys.path.append('../..')
import yaml
import h5py
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
from models import Multi_RLD_Net
import get_f1
import get_matrix
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=50, help='epoch to run[default: 50]')
parser.add_argument('--batch_size', type=int, default=4080, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate[default: 3e-4]')
parser.add_argument('--data_path', default='data/DynamicEarthNet/', help='dataset path')
parser.add_argument('--save_path', default='best_model/', help='model param path')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')
parser.add_argument('--adjust_lr', type=bool, default=True, help='adjust learning rate')
parser.add_argument('--learning_rate_steps', type=list, default=[10,15,20,25,30,35,40,45], help='learning_rate_steps')
parser.add_argument('--learning_rate_gamma', type=float, default=0.5, help='learning_rate_gamma ')
flags = parser.parse_args()
class Trainer(object):
    def __init__(self, cfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.001
        self.model = Multi_RLD_Net.ConvDisRNN(batch_size=flags.batch_size, in_channels=4, out_channels=64, kernel_size=3, num_classes1=2, num_classes2=6, time_len=6).to(self.device)
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

        print('Training..................')
        best_model_pth = '%s/4096_Dynamic_multi_OLDNet_0.6.pth' % (flags.save_path)
        self.model.load_state_dict(torch.load(best_model_pth))
        with h5py.File(flags.data_path + 'test_sample3.h5', 'r') as hf1:
            test_sample = hf1['large_matrix'][:]
        test_sample = torch.from_numpy(test_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])
        print('test_sample', test_sample.shape)
        test_size = test_sample.shape[1]
        test_idx = np.arange(0, test_size)
        iter_test_epoch = test_size // flags.batch_size

        bi_pred = []
        multi_pred = []
        for _iter in range(iter_test_epoch):
            start_idx = _iter * flags.batch_size
            end_idx = (_iter + 1) * flags.batch_size
            batch_val = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_val = torch.as_tensor(batch_val, dtype=torch.float32)
            batch_val = batch_val.to(self.device)

            bi_pred_list, multi_pred_list = self.change_map_epoch(batch_val)
            bi_pred.extend(bi_pred_list)
            multi_pred.extend(multi_pred_list)

        bi_pred_label = np.array(bi_pred)
        multi_pred_label = np.array(multi_pred)
        bi_map = np.reshape(bi_pred_label, (flags.batch_size, flags.batch_size))
        multi_map = np.reshape(multi_pred_label, (flags.batch_size, flags.batch_size))
        bi_map = np.asarray(bi_map)
        multi_map = np.asarray(multi_map)
        io.imsave('Dynamic_multi_bi_OLDNet_3.tif', bi_map)
        io.imsave('Dynamic_multi_multi_OLDNet_3.tif', multi_map)
        plt.imshow(bi_map)
        plt.show()
        plt.imshow(multi_map)
        plt.show()

    def change_map_epoch(self, batch_train):
        self.model.eval()
        bi_pred_list,multi_pred_list = [], []
        self.optim.zero_grad()

        pred1, pred2, out1, pred3, pred4, out2 = self.model(batch_train)
        bi_pred_list += out1.data.max(1)[1].data.cpu().numpy().tolist()
        multi_pred_list += out2.data.max(1)[1].data.cpu().numpy().tolist()
        return bi_pred_list, multi_pred_list

    def test(self):
        print('Training..................')
        best_model_pth = '%s/Dynamic_Multi_RLD_Net.pth' % (flags.save_path)
        self.model.load_state_dict(torch.load(best_model_pth))

        with h5py.File(flags.data_path + 'test_sample.h5', 'r') as hf1:
            test_sample = hf1['large_matrix'][:]
        with h5py.File(flags.data_path + 'test_bi_label.h5', 'r') as hf2:
            test_bi_label = hf2['large_matrix'][:]
        with h5py.File(flags.data_path + 'test_multi_label.h5', 'r') as hf3:
            test_multi_label = hf3['large_matrix'][:]

        test_sample = torch.from_numpy(test_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])
        test_size = test_bi_label.shape[0]
        print('test_sample', test_sample.shape)

        test_idx = np.arange(0, test_size)
        iter_test_epoch = test_size // flags.batch_size

        bi_pred = []
        bi_real = []
        multi_pred = []
        multi_real = []

        for _iter in range(iter_test_epoch):
            start_idx = _iter * flags.batch_size
            end_idx = (_iter + 1) * flags.batch_size
            batch_test_sample = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_test_bi_label = test_bi_label[test_idx[start_idx:end_idx]]
            batch_test_multi_label = test_multi_label[test_idx[start_idx:end_idx]]
            batch_test_sample = torch.as_tensor(batch_test_sample, dtype=torch.float32).to(self.device)
            batch_test_bi_label = torch.as_tensor(batch_test_bi_label, dtype=torch.long).to(self.device)
            batch_test_multi_label = torch.as_tensor(batch_test_multi_label, dtype=torch.long).to(self.device)

            pred_bi_list, target_bi_list, pred_multi_list, target_multi_list = self.test_epoch(batch_test_sample, batch_test_bi_label, batch_test_multi_label)
            bi_pred.extend(pred_bi_list)
            bi_real.extend(target_bi_list)
            multi_pred.extend(pred_multi_list)
            multi_real.extend(target_multi_list)
        print('bi_pred', len(bi_pred))
        print('bi_real', len(bi_real))
        print('multi_pred', len(multi_pred))
        print('multi_real', len(multi_real))
        test_pred_conf = np.asarray(bi_pred)
        test_target_conf = np.asarray(bi_real)
        Test_Matrix = get_matrix.matrix(test_pred_conf, test_target_conf)
        Test_Matrix = np.asarray(Test_Matrix)

        Test_Accuracy, Test_Precision, Test_Recall, Test_F1, Test_Kappa, Test_MIoU, Test_FPR, Test_FNR = get_matrix.accuracy(Test_Matrix)
        print('Test_Matrix:', Test_Matrix)
        print("test_accuracy：%.2f test_f1：%.2f test_kappa：%.2f test_precision：%.2f test_recall：%.2f test_FPR：%.2f test_FNR：%.2f" % (
                Test_Accuracy * 100, Test_F1 * 100, Test_Kappa * 100, Test_Precision * 100, Test_Recall * 100,
                Test_FPR * 100, Test_FNR * 100))

        multi_Marix = (confusion_matrix(multi_real, multi_pred))
        multi_precision = precision_score(multi_real, multi_pred, average='macro')
        multi_f1, multi_OA, multi_avg_F1, multi_Kappa, multi_avg_PA,multi_avg_UA,multi_avg_FPR, multi_avg_FNR = get_f1.test_precision(multi_Marix)
        print('multi_OA: %.4f  multi_avg_F1: %.4f  multi_Kappa: %.4f multi_avg_PA: %.4f   multi_avg_UA: %.4f multi_avg_FPR: %.4f   multi_avg_FNR: %.4f ' % (
         multi_OA, multi_avg_F1, multi_Kappa, multi_avg_PA, multi_avg_UA,multi_avg_FPR, multi_avg_FNR))
        print('multi_precision: %.4f' % (multi_precision))

    def test_epoch(self, batch_train, batch_bi_label, batch_multi_label):
        self.model.eval()
        loss_list, pred_bi_list, target_bi_list, pred_multi_list, target_multi_list = [], [], [], [], []
        self.optim.zero_grad()

        pred1, pred2, out1, pred3, pred4, out2 = self.model(batch_train)
        pred_bi_list = out1.data.max(1)[1].data.cpu().numpy().tolist()
        target_bi_list += batch_bi_label.data.cpu().numpy().tolist()
        pred_multi_list = out2.data.max(1)[1].data.cpu().numpy().tolist()
        target_multi_list += batch_multi_label.data.cpu().numpy().tolist()
        return pred_bi_list, target_bi_list, pred_multi_list, target_multi_list

if __name__ == '__main__':
    trainer = Trainer(flags)
    trainer.test()
    # trainer.change_map()  # get the change map
