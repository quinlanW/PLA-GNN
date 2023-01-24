'''
Train
'''
import os
import json
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import *
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
from torch.nn import MultiLabelSoftMarginLoss


def protein_loc_correction(loc_proba, alpha):  #  24041 12
    """
    Normalization

    :param loc_proba: Position prediction probability matrix
    :param alpha: Coefficient
    :return:
        Location prediction matrix
    """
    min_proba = loc_proba.min(dim=0).values  # 1 12
    max_proba = loc_proba.max(dim=0).values  # 1 12
    new_proba = (loc_proba - min_proba) / (max_proba - min_proba)
    sum_proba = new_proba.sum(dim=1).reshape(-1, 1)  # 24041 1
    new_proba = new_proba / sum_proba

    loc_pred = torch.zeros(loc_proba.shape)
    thresholds = new_proba.max(dim=1).values - (new_proba.max(dim=1).values - new_proba.min(dim=1).values) * alpha
    for row in range(len(loc_proba)):
        threshold = thresholds[row]
        loc_pred[row][new_proba[row] > threshold] = 1.
    loc_pred = loc_pred.double()
    return loc_pred


def performances_record(loc_true, loc_pred):
    """
    Performances record

    :param loc_true: Real location matrix
    :param loc_pred: Predictive location matrix
    :return:
        performance - aim, cov, acc, atr, afr
    """
    loc_true = loc_true.clone().detach().long().cpu()
    loc_pred = loc_pred.clone().detach().long().cpu()
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


def multi_loss(input, target, i_weight):
    """
    Loss function

    :param input: predict
    :param target: real
    :param i_weight: weight
    :return:
        loss
    """
    loss = 0
    for i in range(len(i_weight)):
        scl_input = input[:, i]
        scl_target = target[:, i]
        scl_loss = (scl_target * torch.log(torch.clamp(scl_input, 1e-9, 10.)) * i_weight[i] + (1 - scl_target) * torch.log(torch.clamp(1 - scl_input, 1e-9, 10.))) / (i_weight[i] + 1) * 2
        scl_loss = -scl_loss.sum() / len(input)
        loss += scl_loss
    loss = loss

    return loss


def weight_cal(loc_mat):
    """
    Calculation of different location weights

    :param loc_mat: real location matrix
    :return:
        weight
    """
    class_num = loc_mat.sum(axis=0)  # num in each class
    sample_num = 0
    for line in loc_mat:  # count num with label
        if line.sum():  # if loc
            sample_num += 1
    i_weight = ((sample_num - class_num) / class_num)

    return i_weight

with open('../data/generate_materials/label_list.json') as f:
    label_map = json.load(f)

def label_mapping(x):
    return label_map[x][0]

def res_mapping(a):
    strs = ''
    x = np.where(a == 1)[0] + 1
    for i in range(len(x)):
        strs += str(x[i]) + ', '
    return strs[0:-2]

def train(g, lr, fold_num, epoch_num, alpha_list, device, path):
    log_write_flag = True

    # nodes features and labels
    features = g.ndata['feat']
    labels = g.ndata['loc']

    epoch = list(range(epoch_num))

    # weight
    loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
    i_weight = weight_cal(loc_mat)

    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)


    # loc with labels - num and scale
    p_label_num = labels.cpu().detach().numpy().astype(int).sum(0)
    p_label_scale = p_label_num / len(label) * 100

    fold_seeds = [12, 22, 32, 42, 52, 62, 72, 82, 92, 100]
    fold = 1
    for fseed in fold_seeds:
        kfold = KFold(n_splits=fold_num, random_state=fseed, shuffle=True)

        train_dict, val_dict = {}, {}

        for alpha in alpha_list:
            fold_flag = 1
            train_d, val_d = {}, {}

            # loss_mse = 0
            # loc_pred = 0
            # loc_logit = 0
            # fold_use = 0

            for train_idx, val_idx in kfold.split(label):

                # model_sage = SAGE(g.ndata['feat'].shape[1], 400, 300, 200)
                # model_mlp = MLP(200, 100, 12)
                #
                # sage_mat = model_sage(g, features)
                # torch.save(sage_mat, '../data/sage_mat.pt')

                model = GNN32(g.ndata['feat'].shape[1], 400, 300, 200, 100, 12).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # index conversion
                train_index = []
                val_index = []
                for idx in train_idx:
                    train_index.append(label[idx])
                for idx in val_idx:
                    val_index.append(label[idx])

                # evaluation indicators
                train_aim_list, train_cov_list, train_acc_list, train_atr_list, train_afr_list = [], [], [], [], []
                val_aim_list, val_cov_list, val_acc_list, val_atr_list, val_afr_list = [], [], [], [], []
                train_loss_list, val_loss_list = [], []

                for e in range(epoch_num):
                    # Forward
                    optimizer.zero_grad()
                    model.train()
                    logits = model(g, features)

                    # Compute loss
                    # Note that you should only compute the losses of the nodes in the training set.
                    train_loss = multi_loss(logits[train_index], labels[train_index], i_weight)
                    train_loss.backward()
                    optimizer.step()
                    model.eval()
                    val_loss = multi_loss(logits[val_index], labels[val_index], i_weight)

                    # Compute prediction
                    pred = protein_loc_correction(logits, alpha=alpha)

                    # Compute accuracy on training/validation
                    train_aim, train_cov, train_acc, train_atr, train_afr = performances_record(labels[train_index], pred[train_index])
                    val_aim, val_cov, val_acc, val_atr, val_afr = performances_record(labels[val_index], pred[val_index])

                    # record

                    train_loss_list.append(train_loss.item())
                    val_loss_list.append(val_loss.item())

                    train_aim_list.append(train_aim)
                    train_cov_list.append(train_cov)
                    train_acc_list.append(train_acc)
                    train_atr_list.append(train_atr)
                    train_afr_list.append(train_afr.item())

                    val_aim_list.append(val_aim)
                    val_cov_list.append(val_cov)
                    val_acc_list.append(val_acc)
                    val_atr_list.append(val_atr)
                    val_afr_list.append(val_afr.item())

                    if e % 5 == 0 or e == (epoch_num - 1):
                        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print('TIME: {}, In epoch {} / fold {} / round {}, learning rate: {:.10f}, alpha: {:.2f}'
                              .format(time, e, fold_flag, fold, lr, alpha))
                        print('tra -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                              .format(train_aim, train_cov, train_acc, train_atr, train_afr, train_loss))
                        print('val -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                              .format(val_aim, val_cov, val_acc, val_atr, val_afr, val_loss))
                        p_pred = pred.cpu().detach().numpy().astype(int)  # pred protein loc
                        p_pred_num = p_pred.sum(0)  # sum for each loc
                        p_pred_scale = p_pred_num / len(p_pred) * 100
                        tplt = "{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\t" \
                               "{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\t" \
                               "{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\t{:.2f}%({:<6})\n"
                        print(tplt.format(p_label_scale[0], p_label_num[0], p_label_scale[1], p_label_num[1],
                                          p_label_scale[2], p_label_num[2], p_label_scale[3], p_label_num[3],
                                          p_label_scale[4], p_label_num[4], p_label_scale[5], p_label_num[5],
                                          p_label_scale[6], p_label_num[6], p_label_scale[7], p_label_num[7],
                                          p_label_scale[8], p_label_num[8], p_label_scale[9], p_label_num[9],
                                          p_label_scale[10], p_label_num[10], p_label_scale[11], p_label_num[11]), end='')
                        print(tplt.format(p_pred_scale[0], p_pred_num[0], p_pred_scale[1], p_pred_num[1],
                                          p_pred_scale[2], p_pred_num[2], p_pred_scale[3], p_pred_num[3],
                                          p_pred_scale[4], p_pred_num[4], p_pred_scale[5], p_pred_num[5],
                                          p_pred_scale[6], p_pred_num[6], p_pred_scale[7], p_pred_num[7],
                                          p_pred_scale[8], p_pred_num[8], p_pred_scale[9], p_pred_num[9],
                                          p_pred_scale[10], p_pred_num[10], p_pred_scale[11], p_pred_num[11]), end='')
                        print('-' * 190)
                        # txt log
                        txt_path = path + 'txt_log.txt'
                        with open(txt_path, 'a') as f:
                            if e == 0:
                                f.write('-' * 190)
                                f.write('\n')
                                f.write('-' * 190)
                                f.write('\n')
                                f.write(
                                    'learning rate:{:.8f}, fold num:{}, epoch num:{}, alpha:{}, device:{}\n'
                                    .format(lr, fold_flag, epoch_num, alpha, device))
                                f.write(tplt.format(p_label_scale[0], p_label_num[0], p_label_scale[1], p_label_num[1],
                                                    p_label_scale[2], p_label_num[2], p_label_scale[3], p_label_num[3],
                                                    p_label_scale[4], p_label_num[4], p_label_scale[5], p_label_num[5],
                                                    p_label_scale[6], p_label_num[6], p_label_scale[7], p_label_num[7],
                                                    p_label_scale[8], p_label_num[8], p_label_scale[9], p_label_num[9],
                                                    p_label_scale[10], p_label_num[10], p_label_scale[11], p_label_num[11]))
                            f.write(tplt.format(p_pred_scale[0], p_pred_num[0], p_pred_scale[1], p_pred_num[1],
                                                p_pred_scale[2], p_pred_num[2], p_pred_scale[3], p_pred_num[3],
                                                p_pred_scale[4], p_pred_num[4], p_pred_scale[5], p_pred_num[5],
                                                p_pred_scale[6], p_pred_num[6], p_pred_scale[7], p_pred_num[7],
                                                p_pred_scale[8], p_pred_num[8], p_pred_scale[9], p_pred_num[9],
                                                p_pred_scale[10], p_pred_num[10], p_pred_scale[11], p_pred_num[11]))

                # store performance record
                train_d[fold_flag] = {
                    'aim': train_aim_list, 'cov': train_cov_list, 'acc': train_acc_list, 'loss': train_loss_list
                    # 'atr': train_atr_list, 'afr': train_afr_list,
                }
                val_d[fold_flag] = {
                    'aim': val_aim_list, 'cov': val_cov_list, 'acc': val_acc_list, 'loss': val_loss_list
                    # 'atr': val_atr_list, 'afr': val_afr_list,
                }
                # store predict matrix
                np.save(path + str(fold) + '_' + str(fold_flag) + '_' + 'loc_logits', logits.clone().detach().float().cpu().numpy())

                # store 10 fold information
                # np.set_printoptions(threshold=np.inf)
                # round, fold, flag, idx, label, pred
                train_label = labels[train_index].clone().detach().float().cpu().numpy()
                train_pred = pred[train_index].clone().detach().float().cpu().numpy()
                train_row = train_label.shape[0]
                train_round = np.array([fold] * train_row).reshape(-1, 1)
                train_fold = np.array([fold_flag] * train_row).reshape(-1, 1)
                train_flag = np.array([0] * train_row).reshape(-1, 1)
                # train_label = np.char.array(train_label.astype(int).astype(str))
                # t_label = train_label[:, 0] + train_label[:, 1] + train_label[:, 2] + train_label[:, 3] + \
                #           train_label[:, 4] + train_label[:, 5] + train_label[:, 6] + train_label[:, 7] + \
                #           train_label[:, 8] + train_label[:, 9] + train_label[:, 10] + train_label[:, 11]
                # train_pred = np.char.array(train_pred.astype(int).astype(str))
                # t_pred = train_pred[:, 0] + train_pred[:, 1] + train_pred[:, 2] + train_pred[:, 3] + \
                #          train_pred[:, 4] + train_pred[:, 5] + train_pred[:, 6] + train_pred[:, 7] + \
                #          train_pred[:, 8] + train_pred[:, 9] + train_pred[:, 10] + train_pred[:, 11]

                val_label = labels[val_index].clone().detach().float().cpu().numpy()
                val_pred = pred[val_index].clone().detach().float().cpu().numpy()
                val_row = val_label.shape[0]
                val_round = np.array([fold] * val_row).reshape(-1, 1)
                val_fold = np.array([fold_flag] * val_row).reshape(-1, 1)
                val_flag = np.array([1] * val_row).reshape(-1, 1)
                # val_label = np.char.array(val_label.astype(int).astype(str))
                # v_label = val_label[:, 0] + val_label[:, 1] + val_label[:, 2] + val_label[:, 3] + \
                #           val_label[:, 4] + val_label[:, 5] + val_label[:, 6] + val_label[:, 7] + \
                #           val_label[:, 8] + val_label[:, 9] + val_label[:, 10] + val_label[:, 11]
                # val_pred = np.char.array(val_pred.astype(int).astype(str))
                # v_pred = val_pred[:, 0] + val_pred[:, 1] + val_pred[:, 2] + val_pred[:, 3] + \
                #          val_pred[:, 4] + val_pred[:, 5] + val_pred[:, 6] + val_pred[:, 7] + \
                #          val_pred[:, 8] + val_pred[:, 9] + val_pred[:, 10] + val_pred[:, 11]

                train_index_to_label = np.array(list(map(label_mapping, train_index))).reshape(-1, 1)
                t_label_to_idx = np.array(list(map(res_mapping, train_label))).reshape(-1, 1)
                t_pred_to_idx = np.array(list(map(res_mapping, train_pred))).reshape(-1, 1)
                train_info = np.concatenate((train_round, train_fold, train_flag, train_index_to_label, t_label_to_idx, t_pred_to_idx), axis=1)

                val_index_to_label = np.array(list(map(label_mapping, val_index))).reshape(-1, 1)
                v_label_to_idx = np.array(list(map(res_mapping, val_label))).reshape(-1, 1)
                v_pred_to_idx = np.array(list(map(res_mapping, val_pred))).reshape(-1, 1)
                val_info = np.concatenate((val_round, val_fold, val_flag, val_index_to_label, v_label_to_idx, v_pred_to_idx), axis=1)

                row_info = np.concatenate((train_info, val_info), axis=0)

                with open(path + '/log.tsv', 'a+') as f:
                    writer = csv.writer(f, delimiter='\t')
                    if log_write_flag:
                        writer.writerow(['round', 'fold', 'flag-t0v1', 'index', 'true label', 'predict label'])
                        writer.writerows(row_info)
                        log_write_flag = False
                    else:
                        writer.writerows(row_info)

                fold_flag = fold_flag + 1

            train_dict[alpha] = train_d
            val_dict[alpha] = val_d

        fig_data = {
            'train': train_dict,
            'validation': val_dict
        }
        with open(path + 'fig_data' + '_' + str(fold) + '.json', 'w') as f:
            json.dump(fig_data, f)
        fold = fold + 1





