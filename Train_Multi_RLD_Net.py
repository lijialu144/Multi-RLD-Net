import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import numpy as np
import torch.nn as nn
from models import Dynamic_multitask_OLDNet
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import get_f1
import h5py
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
np.random.seed(1)

def FWCLoss(pred,target, count_pos, count_neg):                                                                                #对预测的概率进行Sigmoid激活
    m = nn.Sigmoid()
    lossinput = m(pred)
    ratio= count_neg/(count_pos + count_neg)                                                              #负样本的比例                                                                  #将1维的展开为2维，相当于复制了一个维度  500变为500*2
    L = - (ratio * target * torch.log(lossinput + 1e-10) * torch.pow(input=(1-lossinput),exponent=2) +
           (1 - ratio) * (1 - target) * torch.log(1 - lossinput + 1e-10) * torch.pow(input=lossinput,exponent=2))
    W = torch.ones(target.shape).cuda()
    output = torch.mean(L * W)
    return output

def multiFWCLoss(pred, target, label):
    batch_size = pred.size(0)
    label_0 = torch.nonzero(label == 0).cuda()
    label_1 = torch.nonzero(label == 1).cuda()
    label_2 = torch.nonzero(label == 2).cuda()
    label_3 = torch.nonzero(label == 3).cuda()
    label_4 = torch.nonzero(label == 4).cuda()
    label_5 = torch.nonzero(label == 5).cuda()

    ratio_0 = ((batch_size - label_0.shape[0]) / batch_size)
    ratio_1 = ((batch_size - label_1.shape[0]) / batch_size)
    ratio_2 = ((batch_size - label_2.shape[0]) / batch_size)
    ratio_3 = ((batch_size - label_3.shape[0]) / batch_size)
    ratio_4 = ((batch_size - label_4.shape[0]) / batch_size)
    ratio_5 = ((batch_size - label_5.shape[0]) / batch_size)

    ratio_0_f = torch.ones([4096, 1]) * ratio_0
    ratio_1_f = torch.ones([4096, 1]) * ratio_1
    ratio_2_f = torch.ones([4096, 1]) * ratio_2
    ratio_3_f = torch.ones([4096, 1]) * ratio_3
    ratio_4_f = torch.ones([4096, 1]) * ratio_4
    ratio_5_f = torch.ones([4096, 1]) * ratio_5

    ratio = torch.cat((ratio_0_f, ratio_1_f, ratio_2_f, ratio_3_f, ratio_4_f, ratio_5_f),dim=1).cuda()
    L = - (ratio * target * torch.log(pred + 1e-10) * torch.pow(input=(1 - pred), exponent=2))
    W = torch.ones(target.shape).cuda()
    output = torch.mean(L * W)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=20, help='epoch')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--patch_size', type=float, default=5, help='patch_size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--data_path', type=str, default='../../data/Dynamic_h5/', help='Fsplit folder')
parser.add_argument('--save_path', default='best_models/Dynamic', help='saved model path')
parser.add_argument('--adjust_lr', type=bool, default=True, help='adjust learning rate')
parser.add_argument('--learning_rate_steps', type=list, default=[10,15,20,25,30,35,40,45], help='learning_rate_steps')
parser.add_argument('--learning_rate_gamma', type=float, default=0.5, help='learning_rate_gamma ')
flags = parser.parse_args()

class Trainer(object):
    def __init__(self, flags):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = flags.learning_rate
        print('======distance rnn========')
        self.model = Dynamic_multitask_OLDNet.ConvDisRNN(batch_size=flags.batch_size, in_channels=4, out_channels=64,
                                                 kernel_size=3, num_classes1=2, num_classes2=6, time_len=6).to(self.device)
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
        with h5py.File(flags.data_path + 'train_sample.h5', 'r') as hf1:
            train_sample = hf1['large_matrix'][:]
        with h5py.File(flags.data_path + 'train_bi_label.h5', 'r') as hf2:
            train_bi_label = hf2['large_matrix'][:]
        with h5py.File(flags.data_path + 'train_multi_label.h5', 'r') as hf3:
            train_multi_label = hf3['large_matrix'][:]
        with h5py.File(flags.data_path + 'val_sample.h5', 'r') as hf4:
            val_sample = hf4['large_matrix'][:]
        with h5py.File(flags.data_path + 'val_bi_label.h5', 'r') as hf5:
            val_bi_label = hf5['large_matrix'][:]
        with h5py.File(flags.data_path + 'val_multi_label.h5', 'r') as hf6:
            val_multi_label = hf6['large_matrix'][:]

        train_sample = torch.from_numpy(train_sample)
        train_sample = train_sample.permute([0, 1, 4, 2, 3])
        val_sample = torch.from_numpy(val_sample)
        val_sample = val_sample.permute([0, 1, 4, 2, 3])
        train_size = train_bi_label.shape[0]
        val_size = val_bi_label.shape[0]
        print('train_sample', train_sample.shape)
        print('val_sample', val_sample.shape)

        train_idx = np.arange(0, train_size)
        np.random.seed(1)
        np.random.shuffle(train_idx)
        val_idx = np.arange(0, val_size)
        np.random.seed(1)
        np.random.shuffle(val_idx)

        best_loss = 2000
        best_f1 = 0
        BATCH_SZ = flags.batch_size
        iter_train_epoch = train_size // BATCH_SZ
        iter_val_epoch = val_size // BATCH_SZ

        model_root = osp.join(flags.save_path)
        train_Loss_list = []
        val_Loss_list = []
        val_Accuracy_list = []

        for epoch in tqdm(range(0, flags.max_epoch)):
            print('self.lr', self.lr)
            train_ave_loss = 0
            train_ave_bi_OA = 0
            train_ave_bi_F1 = 0
            train_ave_multi_OA = 0
            train_ave_multi_F1 = 0
            train_ave_multi_Kappa = 0
            val_ave_loss = 0
            val_ave_bi_OA = 0
            val_ave_bi_F1 = 0
            val_ave_multi_OA = 0
            val_ave_multi_F1 = 0
            val_ave_multi_Kappa = 0
            if flags.adjust_lr:
                self.adjust_learning_rate(self.optim, epoch, flags.learning_rate_steps, flags.learning_rate_gamma)
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

            for _iter in range(iter_train_epoch):
                start_idx = _iter * BATCH_SZ
                end_idx = (_iter + 1) * BATCH_SZ
                batch_train_sample = train_sample[:, train_idx[start_idx:end_idx], :, :, :]
                batch_train_bi_label = train_bi_label[train_idx[start_idx:end_idx]]
                batch_train_multi_label = train_multi_label[train_idx[start_idx:end_idx]]

                batch_train_sample = torch.as_tensor(batch_train_sample, dtype=torch.float32).to(self.device)
                batch_train_bi_label = torch.as_tensor(batch_train_bi_label, dtype=torch.long).to(self.device)
                batch_train_multi_label = torch.as_tensor(batch_train_multi_label, dtype=torch.long).to(self.device)

                train_epoch_loss, train_bi_OA, train_bi_F1, train_multi_OA, train_multi_F1, train_multi_Kappa = \
                    self.train_epoch(batch_train_sample, batch_train_bi_label, batch_train_multi_label)

                train_ave_loss += train_epoch_loss
                train_ave_bi_OA += train_bi_OA
                train_ave_bi_F1 += train_bi_F1
                train_ave_multi_OA += train_multi_OA
                train_ave_multi_F1 += train_multi_F1
                train_ave_multi_Kappa += train_multi_Kappa

            for _iter in range(iter_val_epoch):
                start_idx = _iter * BATCH_SZ
                end_idx = (_iter + 1) * BATCH_SZ
                batch_val_sample = val_sample[:, val_idx[start_idx:end_idx], :, :, :]
                batch_val_bi_label = val_bi_label[val_idx[start_idx:end_idx]]
                batch_val_multi_label = val_multi_label[val_idx[start_idx:end_idx]]

                batch_val_sample = torch.as_tensor(batch_val_sample, dtype=torch.float32).to(self.device)
                batch_val_bi_label = torch.as_tensor(batch_val_bi_label, dtype=torch.long).to(self.device)
                batch_val_multi_label = torch.as_tensor(batch_val_multi_label, dtype=torch.long).to(self.device)

                val_epoch_loss, val_bi_OA, val_bi_F1, val_multi_OA, val_multi_F1, val_multi_Kappa\
                    = self.val_epoch(batch_val_sample, batch_val_bi_label, batch_val_multi_label)
                val_ave_loss += val_epoch_loss
                val_ave_bi_OA += val_bi_OA
                val_ave_bi_F1 += val_bi_F1
                val_ave_multi_OA += val_multi_OA
                val_ave_multi_F1 += val_multi_F1
                val_ave_multi_Kappa += val_multi_Kappa

            best_model1 = flags.save_path + '/multi_OLDNet_model_' + str(epoch) + '.pth'
            # best_model2 = '%s/best_model_02.pth' % (model_root)
            # best_model_last2 = '%s/last_model_02.pth' % (model_root)
            # best_model_last1 = '%s/last_model_01.pth' % (model_root)

            train_ave_loss /= iter_train_epoch
            train_ave_bi_OA /= iter_train_epoch
            train_ave_bi_F1 /= iter_train_epoch
            train_ave_multi_OA /= iter_train_epoch
            train_ave_multi_F1 /= iter_train_epoch
            train_ave_multi_Kappa /= iter_train_epoch

            val_ave_loss /= iter_val_epoch
            val_ave_bi_OA /= iter_val_epoch
            val_ave_bi_F1 /= iter_val_epoch
            val_ave_multi_OA /= iter_val_epoch
            val_ave_multi_F1 /= iter_val_epoch
            val_ave_multi_Kappa /= iter_val_epoch

            train_Loss_list.append(train_ave_loss)
            val_Loss_list.append(val_ave_loss)
            print("train_ave_loss：%.4f  train_ave_bi_OA:%.4f  train_ave_bi_F1:%.4f  train_ave_multi_OA%.4f  train_ave_multi_F1%.4f  "
                  "train_ave_multi_Kappa：%.4f  " % (
                train_ave_loss, train_ave_bi_OA, train_ave_bi_F1, train_ave_multi_OA, train_ave_multi_F1, train_ave_multi_Kappa))
            print("val_ave_loss：%.4f  val_ave_bi_OA：%.4f  val_ave_bi_F1：%.4f  val_ave_multi_OA：%.4f  val_ave_multi_F1：%.4f  "
                  "val_ave_multi_Kappa：%.4f" % (
                val_ave_loss, val_ave_bi_OA, val_ave_bi_F1, val_ave_multi_OA, val_ave_multi_F1, val_ave_multi_Kappa))

            torch.save(self.model.state_dict(), best_model1)
        #     if val_ave_loss < best_loss:
        #         best_loss = val_ave_loss
        #         torch.save(self.model.state_dict(), best_model1)
        #         print("best loss model is saved")
        #     if val_ave_f1 > best_f1:
        #         best_f1 = val_ave_f1
        #         torch.save(self.model.state_dict(), best_model2)
        #         print("best f1 model is saved")
        #     if (flags.max_epoch-epoch)==2:
        #         torch.save(self.model.state_dict(), best_model_last2)
        #         print("last second model is saved")
        #
        #     if (flags.max_epoch-epoch)==1:
        #         torch.save(self.model.state_dict(), best_model_last1)
        #         print("last first model is saved")
        # plt.title('train loss & val loss')
        # plt.plot(train_Loss_list)
        # plt.plot(val_Loss_list)
        # plt.savefig('best_models/UTRNet/loss.png')
        # plt.show()

    def train_epoch(self, batch_train, batch_bi_label, batch_multi_label):
        self.model.train()
        pred_bi_list, target_bi_list, pred_multi_list, target_multi_list, loss_list = [], [], [], [], []
        self.optim.zero_grad()
        pred1, pred2, out1, pred3, pred4, out2 = self.model(batch_train)
        count_pos = torch.sum(batch_bi_label)
        count_neg = torch.sum(1 - batch_bi_label)
        target1 = torch.eye(2)[batch_bi_label, :].to(self.device)
        target2 = torch.eye(6)[batch_multi_label, :].to(self.device)

        loss1 = FWCLoss(pred1, target1, count_pos, count_neg)
        loss2 = FWCLoss(pred2, target1, count_pos, count_neg)
        loss3 = FWCLoss(out1, target1, count_pos, count_neg)
        loss4 = multiFWCLoss(pred3, target2, batch_multi_label)
        loss5 = multiFWCLoss(pred4, target2, batch_multi_label)
        loss6 = multiFWCLoss(out2, target2, batch_multi_label)
        loss = loss1 + loss2 + loss3 + 0.5*(loss4 + loss5 +loss6)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
        self.optim.step()

        pred_bi_list += out1.data.max(1)[1].data.cpu().numpy().tolist()
        target_bi_list += batch_bi_label.data.cpu().numpy().tolist()
        pred_multi_list += out2.data.max(1)[1].data.cpu().numpy().tolist()
        target_multi_list += batch_multi_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        train_epoch_loss = loss.item()
        bi_OA = accuracy_score(target_bi_list, pred_bi_list)
        bi_F1 = f1_score(target_bi_list, pred_bi_list, pos_label=1)
        multi_Marix = (confusion_matrix(target_multi_list, pred_multi_list))
        multi_OA, multi_F1, multi_Kappa = get_f1.train_precision(multi_Marix)
        return train_epoch_loss, bi_OA, bi_F1, multi_OA, multi_F1, multi_Kappa

    def val_epoch(self, batch_val, batch_bi_label, batch_multi_label):
        self.model.eval()
        pred_bi_list, target_bi_list, pred_multi_list, target_multi_list, loss_list = [], [], [], [], []
        self.optim.zero_grad()

        pred1, pred2, out1, pred3,pred4, out2 = self.model(batch_val)
        count_pos = torch.sum(batch_bi_label)
        count_neg = torch.sum(1 - batch_bi_label)
        target1 = torch.eye(2)[batch_bi_label, :].to(self.device)
        target2 = torch.eye(6)[batch_multi_label, :].to(self.device)

        loss1 = FWCLoss(pred1, target1, count_pos, count_neg)
        loss2 = FWCLoss(pred2, target1, count_pos, count_neg)
        loss3 = FWCLoss(out1, target1, count_pos, count_neg)
        loss4 = multiFWCLoss(pred3, target2, batch_multi_label)
        loss5 = multiFWCLoss(pred4, target2, batch_multi_label)
        loss6 = multiFWCLoss(out2, target2, batch_multi_label)
        loss = loss1 + loss2 + loss3 + 0.5*(loss4 + loss5 + loss6)

        pred_bi_list += out1.data.max(1)[1].data.cpu().numpy().tolist()
        target_bi_list += batch_bi_label.data.cpu().numpy().tolist()
        pred_multi_list += out2.data.max(1)[1].data.cpu().numpy().tolist()
        target_multi_list += batch_multi_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        val_epoch_loss = loss.item()
        bi_OA = accuracy_score(target_bi_list, pred_bi_list)
        bi_F1 = f1_score(target_bi_list, pred_bi_list, pos_label=1)
        multi_Marix = (confusion_matrix(target_multi_list, pred_multi_list))
        multi_OA, multi_F1, multi_Kappa = get_f1.train_precision(multi_Marix)
        return val_epoch_loss, bi_OA, bi_F1, multi_OA, multi_F1, multi_Kappa

if __name__ == '__main__':
    trainer = Trainer(flags)
    trainer.train()
