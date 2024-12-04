import os
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import torchnet as tnt
from skimage import io
import cv2


def matrix(pred, ch_target):

    bi_result = np.reshape(pred, [-1, 1])
    ref_change = np.reshape(ch_target, [-1, 1])

    test_change_num = np.sum(ref_change[ref_change == 1])
    test_unchange_num = len(ref_change[ref_change == 0])
    test_num = test_change_num + test_unchange_num

    TP = np.sum((bi_result * ref_change) == 1)
    TN = np.sum(((1 - bi_result) * (1 - ref_change)) == 1)
    FN = test_change_num - TP
    FP = test_unchange_num - TN
    Matrix = [[TP, FP], [FN, TN]]

    return Matrix

def accuracy(matrix):
    TP = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TN = matrix[1, 1]
    Matrix = [[TP, FP], [FN, TN]]
    test_num = TP + FP + FN + TN

    OA = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    p1 = np.int(np.int(np.int(TP + FP) * np.int(TP + FN)) + np.int(np.int(FN + TN) * np.int(FP + TN)))
    Pe = np.float(p1 / test_num / test_num)
    kappa = (OA - Pe) / (1 - Pe)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = 1 - FPR
    BA = (TPR + TNR) / 2
    Iou = TP / (FN + FP + TP)
    MIoU = (TP / (FN + FP + TP) + TN / (TN + FN + FP)) / 2

    FPR = FP / (FP + TN)
    FNR = 1 - recall

    return  OA, precision, recall, F1, kappa, MIoU, FPR, FNR

def conf_m(output, target_th):
    #作用是将数据转换为以为的向量
    # print('\noutput', output.shape)
    # print('target_th', target_th.shape)
    output_conf=output.data
    # print('output_conf',output_conf.shape)
    output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2))
    # print('output_conf',output_conf.shape)
    target_conf=target_th.data
    # print('target_conf',target_conf.shape)
    target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))
    # print('target_conf',target_conf.shape)
    return output_conf, target_conf

