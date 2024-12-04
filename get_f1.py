import numpy as np
def train_precision(train_Matrix):
    recall_0 = train_Matrix[0, 0] / train_Matrix[0].sum()
    recall_1 = train_Matrix[1, 1] / train_Matrix[1].sum()
    recall_2 = train_Matrix[2, 2] / train_Matrix[2].sum()
    recall_3 = train_Matrix[3, 3] / train_Matrix[3].sum()
    recall_4 = train_Matrix[4, 4] / train_Matrix[4].sum()
    recall_5 = train_Matrix[5, 5] / train_Matrix[5].sum()

    precision_0 = train_Matrix[0, 0] / train_Matrix[:, 0].sum()
    precision_1 = train_Matrix[1, 1] / train_Matrix[:, 1].sum()
    precision_2 = train_Matrix[2, 2] / train_Matrix[:, 2].sum()
    precision_3 = train_Matrix[3, 3] / train_Matrix[:, 3].sum()
    precision_4 = train_Matrix[4, 4] / train_Matrix[:, 4].sum()
    precision_5 = train_Matrix[5, 5] / train_Matrix[:, 5].sum()

    precision = [precision_0, precision_1, precision_2, precision_3, precision_4, precision_5]
    recall = [recall_0, recall_1, recall_2, recall_3, recall_4, recall_5]
    precision = np.nan_to_num(precision, nan=0)
    recall = np.nan_to_num(recall, nan=0)

    f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1, nan=0)

    oa_sum = (train_Matrix[0, 0] + train_Matrix[1, 1] + train_Matrix[2, 2] +
              train_Matrix[3, 3] + train_Matrix[4, 4] + train_Matrix[5, 5])/train_Matrix.sum()
    f1_avg = f1.mean()
    sum = train_Matrix.sum()
    pe = ((train_Matrix[0].sum() * train_Matrix[:, 0].sum()) + (train_Matrix[1].sum() * train_Matrix[:, 1].sum()) +\
         (train_Matrix[2].sum() * train_Matrix[:, 2].sum()) + (train_Matrix[3].sum() * train_Matrix[:, 3].sum()) +\
         (train_Matrix[4].sum() * train_Matrix[:, 4].sum()) + (train_Matrix[5].sum() * train_Matrix[:, 5].sum()))/(train_Matrix.sum()*train_Matrix.sum())
    kappa = (oa_sum-pe)/(1-pe)
    return oa_sum, f1_avg, kappa

def test_precision(test_Matrix):
    recall_0 = test_Matrix[0, 0] / test_Matrix[0].sum()
    recall_1 = test_Matrix[1, 1] / test_Matrix[1].sum()
    recall_2 = test_Matrix[2, 2] / test_Matrix[2].sum()
    recall_3 = test_Matrix[3, 3] / test_Matrix[3].sum()
    recall_4 = test_Matrix[4, 4] / test_Matrix[4].sum()
    recall_5 = test_Matrix[5, 5] / test_Matrix[5].sum()

    precision_0 = test_Matrix[0, 0] / test_Matrix[:, 0].sum()
    precision_1 = test_Matrix[1, 1] / test_Matrix[:, 1].sum()
    precision_2 = test_Matrix[2, 2] / test_Matrix[:, 2].sum()
    precision_3 = test_Matrix[3, 3] / test_Matrix[:, 3].sum()
    precision_4 = test_Matrix[4, 4] / test_Matrix[:, 4].sum()
    precision_5 = test_Matrix[5, 5] / test_Matrix[:, 5].sum()

    FNR_0 = 1 - recall_0
    FNR_1 = 1 - recall_1
    FNR_2 = 1 - recall_2
    FNR_3 = 1 - recall_3
    FNR_4 = 1 - recall_4
    FNR_5 = 1 - recall_5

    FP_0 = test_Matrix[:, 0].sum() - test_Matrix[0, 0]
    FP_1 = test_Matrix[:, 1].sum() - test_Matrix[1, 1]
    FP_2 = test_Matrix[:, 2].sum() - test_Matrix[2, 2]
    FP_3 = test_Matrix[:, 3].sum() - test_Matrix[3, 3]
    FP_4 = test_Matrix[:, 4].sum() - test_Matrix[4, 4]
    FP_5 = test_Matrix[:, 5].sum() - test_Matrix[5, 5]

    TN_0 = test_Matrix.sum() - test_Matrix[:, 0] - test_Matrix[0, :] + test_Matrix[0, 0]
    TN_1 = test_Matrix.sum() - test_Matrix[:, 1] - test_Matrix[1, :] + test_Matrix[1, 1]
    TN_2 = test_Matrix.sum() - test_Matrix[:, 2] - test_Matrix[2, :] + test_Matrix[2, 2]
    TN_3 = test_Matrix.sum() - test_Matrix[:, 3] - test_Matrix[3, :] + test_Matrix[3, 3]
    TN_4 = test_Matrix.sum() - test_Matrix[:, 4] - test_Matrix[4, :] + test_Matrix[4, 4]
    TN_5 = test_Matrix.sum() - test_Matrix[:, 5] - test_Matrix[5, :] + test_Matrix[5, 5]

    FPR_0 = FP_0 / (FP_0 + TN_0)
    FPR_1 = FP_1 / (FP_1 + TN_1)
    FPR_2 = FP_2 / (FP_2 + TN_2)
    FPR_3 = FP_3 / (FP_3 + TN_3)
    FPR_4 = FP_4 / (FP_4 + TN_4)
    FPR_5 = FP_5 / (FP_5 + TN_5)


    precision = [precision_0, precision_1, precision_2, precision_3, precision_4, precision_5]
    recall = [recall_0, recall_1, recall_2, recall_3, recall_4, recall_5]
    FPR = [FPR_0, FPR_1, FPR_2, FPR_3, FPR_4, FPR_5]
    FNR = [FNR_0, FNR_1, FNR_2, FNR_3, FNR_4, FNR_5]
    precision = np.nan_to_num(precision, nan=0)
    recall = np.nan_to_num(recall, nan=0)
    FPR = np.nan_to_num(FPR, nan=0)
    FNR = np.nan_to_num(FNR, nan=0)
    f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1, nan=0)

    oa_sum = (test_Matrix[0, 0] + test_Matrix[1, 1] + test_Matrix[2, 2] + test_Matrix[3, 3] + test_Matrix[4, 4] +
         test_Matrix[5, 5])/test_Matrix.sum()

    PA_avg = precision.mean()
    UA_avg = recall.mean()
    f1_avg = f1.mean()
    FPR_avg = FPR.mean()
    FNR_avg = FNR.mean()
    sum = test_Matrix.sum()
    pe = ((test_Matrix[0].sum() * test_Matrix[:, 0].sum()) + (test_Matrix[1].sum() * test_Matrix[:, 1].sum()) + \
          (test_Matrix[2].sum() * test_Matrix[:, 2].sum()) + (test_Matrix[3].sum() * test_Matrix[:, 3].sum()) + \
          (test_Matrix[4].sum() * test_Matrix[:, 4].sum()) + (test_Matrix[5].sum() * test_Matrix[:, 5].sum())) / (
                     test_Matrix.sum() * test_Matrix.sum())
    kappa = (oa_sum - pe) / (1 - pe)
    return f1,oa_sum, f1_avg, kappa, PA_avg, UA_avg, FPR_avg, FNR_avg