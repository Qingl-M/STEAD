DATASET1 = r"merge_trace_metric2.csv"
DATASET2 = r"merge_trace_metric_label2test.csv"

import pandas as pd
import json
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score
train_data = pd.read_csv(DATASET1, header=[0,1])
raw_test_data = pd.read_csv(DATASET2, header=[0,1])
train_data = train_data.drop(['timestamp'], axis = 1)
train_data.columns.names = ['service','metric']
tempm_train = train_data.swaplevel('metric','service',axis=1).stack()
tempm_train = (tempm_train-tempm_train.mean())/(tempm_train.std())
train_data = tempm_train.unstack().swaplevel('metric','service',axis=1).stack().unstack()
test_data = raw_test_data.drop(['timestamp', 'label'], axis = 1)
test_data.columns.names = ['service','metric']
tempm_test = test_data.swaplevel('metric','service',axis=1).stack()
tempm_test = (tempm_test-tempm_test.mean())/(tempm_test.std())
test_data = tempm_test.unstack().swaplevel('metric','service',axis=1).stack().unstack()
test_data.to_csv(r"test1_normalize.csv")
edge_index =[[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 5, 5, 6, 6, 6, 10, 11, 3, 3, 3, 3, 3],
            [0, 1, 2, 4, 6, 7, 10, 11, 1, 4, 5, 4, 5, 10, 6, 8, 9, 10, 11, 0, 1, 3, 5, 10]]
RESULT="../topomad-1-results/MMS_GAT"
NAME="performance_evaluation_GRU_win30"

def calculate_metrics(pred, ground_truth):
    max_f1 = 0
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    for threshold in pred:
        pred_labels = [1 if pred_i >= threshold else 0 for pred_i in pred]
        accuracy = accuracy_score(ground_truth, pred_labels)
        precision = precision_score(ground_truth, pred_labels)
        f1 = f1_score(ground_truth, pred_labels)
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
    return best_threshold, best_accuracy, best_precision, max_f1

import pandas as pd
from model_main import DG_GRU_VAE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pickle
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_choice =1
seq_len = 20
num_epo = 50
hid_dim = 3
lea_rat = 1e-4
nod_num = 12
sam_num = 20
del_tol = 5
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, f1_score
from STE_GRU_En import GraphGRU_VAE
import pickle
traintimes = []
record_losses = []
for seedi in range(5):
    aps = []
    maps = []
    testtimes = []
    best_thresholds=[]
    best_accuracys=[]
    best_precisions=[]
    max_f1s=[]
    model = GraphGRU_VAE(name=NAME + "VAE", sequence_length=seq_len, num_epochs=num_epo, hidden_dim=hid_dim,
                             lr=lea_rat, gpu=gpu_choice, variational=True, kind='GAT', seed=seedi, head=(8, 8),
                             dropout=(0.4, 0.4), bias=(True, False))
    traintime_start = time.time()
    model.fit(train_data, nod_num, edge_index)
    traintime_end = time.time()
    traintimes.append(traintime_end - traintime_start)
    ground_truth1 = raw_test_data['label']
    ground_truth1.columns = ['label']
    ground_truth = ground_truth1['label'].tolist()
    testtime_start = time.time()
    pred = model.predict(test_data, nod_num, edge_index, sam_num, del_tol)[1]
    testtime_end = time.time()
    testtimes.append(testtime_end - testtime_start)
    preds, errs_system, scores_ori_node = model.predict(test_data, nod_num, edge_index, sam_num, del_tol)
    print(len(raw_test_data['label'][seq_len - del_tol:]))
    f1_scores = np.zeros(len(preds))
    precision, recall, thresholds = precision_recall_curve(raw_test_data['label'][seq_len - del_tol:], preds)
    for i in range(len(precision)-1):
        if precision[i] == 0 or recall[i] == 0:
            f1_scores[i] = 0.5
        else:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    index = np.argmax(f1_scores)

    print('Precision: ', precision[index])
    print('Recall: ', recall[index])
    print('F1 score: ', f1_scores[index])










