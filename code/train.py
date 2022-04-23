import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from model import *
from sklearn.model_selection import KFold
from scipy.sparse import load_npz
from torch.nn import MultiLabelSoftMarginLoss, BCELoss, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score


def protein_loc_correction(loc_proba, alpha):
    min_proba = loc_proba.min(dim=1).values.reshape(len(loc_proba), 1)
    max_proba = loc_proba.max(dim=1).values.reshape(len(loc_proba), 1)
    new_proba = (loc_proba - min_proba) / (max_proba - min_proba)
    sum_proba = new_proba.sum(dim=1).reshape(len(new_proba), 1)
    new_proba = new_proba / sum_proba

    loc_pred = torch.zeros(loc_proba.shape)
    thresholds = new_proba.max(dim=1).values - (new_proba.max(dim=1).values - new_proba.min(dim=1).values) * alpha
    for row in range(len(loc_proba)):
        threshold = thresholds[row]
        loc_pred[row][new_proba[row] > threshold] = 1.
    loc_pred = loc_pred.double()
    return loc_pred


def proformances_record(loc_true, loc_pred):
    # will chage data?
    loc_true = torch.tensor(loc_true.clone().detach().long(), device='cpu')
    loc_pred = torch.tensor(loc_pred.clone().detach().long(), device='cpu')
    mask = torch.tensor([1] * len(loc_true[0]), device='cpu')
    aim = 0.
    cov = 0.
    acc = 0.
    atr = 0.
    afr = 0.
    for i in range(len(loc_true)):
        loc_true[i] = torch.eq(mask, loc_true[i])
        loc_pred[i] = torch.eq(mask, loc_pred[i])
        and_set = (loc_true[i] & loc_pred[i]).sum().float()
        pred = loc_pred[i].sum().float()
        real = loc_true[i].sum().float()
        or_set = (loc_true[i] | loc_pred[i]).sum().float()
        correct = 0
        flag = torch.eq(loc_true[i], loc_pred[i])
        if torch.all(flag):
            correct = 1
        if pred == 0:
            aim = aim + 0
        else:
            aim = aim + and_set / pred
        cov = cov + and_set / real
        acc = acc + and_set / or_set
        atr = atr + correct
        afr = afr + (or_set - and_set) / len(loc_true[i])

    aim = float(aim / len(loc_true))
    cov = float(cov / len(loc_true))
    acc = float(acc / len(loc_true))
    atr = atr / len(loc_true)
    afr = afr / len(loc_true)

    return aim, cov, acc, atr, afr


def weight_cal(loc_mat, beta):
    class_num = loc_mat.sum(axis=0)
    sample_num = 0
    for line in loc_mat:
        if line.sum():
            sample_num += 1
    # weight between classes
    b_weight = sample_num / (class_num * 12)
    # weight inner classes
    i_weight = (sample_num - class_num) / class_num * beta

    return b_weight, i_weight


def train(g, lr, fold_num, epoch_num, alpha_list, beta_list, device):
    # nodes features and labels
    features = g.ndata['feat']
    labels = g.ndata['loc']

    for beta in beta_list:
        # mkdir
        beta_path = '../data/log/b' + str(int(beta * 10)) + '/'
        if not os.path.exists(beta_path):
            os.mkdir(beta_path)
        # weight
        loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
        b_weight, i_weight = weight_cal(loc_mat, beta=beta)

        # loss setting
        criterion = BCELoss()

        # loading files (for label and mask)
        with open('../data/generate_materials/label_with_loc_list.json') as f:
            label = json.load(f)
        with open('../data/generate_materials/label_with_fig.json') as f:
            fig_label = json.load(f)

        # loc with labels - num and scale
        p_label_num = labels.cpu().detach().numpy().astype(int).sum(0)
        p_label_scale = p_label_num / len(label) * 100

        kfold = KFold(n_splits=fold_num, random_state=42, shuffle=True)

        for alpha in alpha_list:
            fold_flag = 1
            for train_idx, val_idx in kfold.split(label):
                model = GNN2(g.ndata['feat'].shape[1], 3000, 500, 1).to(device)
                # model = GNN1(g.ndata['feat'].shape[1], 1000, 12).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # index conversion
                train_index = []
                val_index = []
                for idx in train_idx:
                    train_index.append(label[idx])
                for idx in val_idx:
                    val_index.append(label[idx])

                for e in range(epoch_num):
                    # Forward
                    logits = model(g, features)

                    # Compute loss
                    # Note that you should only compute the losses of the nodes in the training set.
                    train_loss = criterion(logits[train_index], labels[train_index][:, -1:])
                    val_loss = criterion(logits[val_index], labels[val_index][:, -1:])

                    # Compute accuracy on training/validation
                    tra_pred = logits[train_index].detach().numpy()
                    tra_true = labels[train_index][:, -1:].detach().numpy()
                    val_pred = logits[val_index].detach().numpy()
                    val_true = labels[val_index][:, -1:].detach().numpy()
                    train_roc = roc_auc_score(tra_true, tra_pred)
                    val_roc = roc_auc_score(val_true, val_pred)

                    # Backward
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    if e % 5 == 0 or e == epoch_num:
                        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print('TIME: {}, In epoch {} /fold {}, learning rate: {:.10f}, alpha: {:.2f}, beta: {:.2f}'
                              .format(time, e, fold_flag, lr, alpha, beta))
                        print('tra -- roc: {:.3f}, loss: {:.8f}'
                              .format(train_roc, train_loss))
                        print('val -- roc: {:.3f}, loss: {:.8f}'
                              .format(val_roc, val_loss))

                        print('-' * 100)






