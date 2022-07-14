'''
Train
'''
import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from model import *
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
from torch.nn import MultiLabelSoftMarginLoss


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
    loss = 0
    for i in range(len(i_weight)):
        scl_input = input[:, i]
        scl_target = target[:, i]
        ## scl_loss = (scl_target * torch.log(scl_input) * i_weight[i] + (1 - scl_target) * torch.log(1 - scl_input)) / (i_weight[i] + 1) * 2
        scl_loss = (scl_target * torch.log(torch.clamp(scl_input, 1e-9, 10.)) * i_weight[i] + (1 - scl_target) * torch.log(torch.clamp(1 - scl_input, 1e-9, 10.))) / (i_weight[i] + 1) * 2
        # scl_loss = scl_target * torch.log(torch.clamp(scl_input, 1e-9, 10.)) + (1 - scl_target) * torch.log(torch.clamp(1 - scl_input, 1e-9, 10.))
        # scl_loss = (scl_target * torch.log(torch.clamp(scl_input, 1e-6, 10.)) * i_weight[i] * ((1 - scl_input) ** 2) + (1 - scl_target) * torch.log(torch.clamp(1 - scl_input, 1e-6, 10.)) * ((1 - scl_input) ** 2)) / (i_weight[i] + 1) * 2
        # scl_loss = scl_target * torch.log(torch.clamp(scl_input, 1e-6, 10.))
        scl_loss = -scl_loss.sum() / len(input)
        loss += scl_loss
    loss = loss # / len(i_weight)

    return loss


def weight_cal(loc_mat, beta):
    class_num = loc_mat.sum(axis=0)  # num in each class
    sample_num = 0
    for line in loc_mat:  # count num with label
        if line.sum():  # if loc
            sample_num += 1
    # weight between classes
    # b_weight = sample_num / (class_num * 12)
    # b_weight = class_num.sum() / (class_num * 12)
    # b_weight = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 目前看来没有类间权重效果更好
    # weight inner classes
    i_weight = ((sample_num - class_num) / class_num)#  * beta
    # print(i_weight)

    return i_weight


def train(g, lr, fold_num, epoch_num, alpha_list, beta, device, path, seed):
    # nodes features and labels
    features = g.ndata['feat']
    labels = g.ndata['loc']

    epoch = list(range(epoch_num))

    # weight
    loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
    # loc_mat = load_npz('../data/generate_materials/loc_matrix_new.npz').toarray()
    i_weight = weight_cal(loc_mat, beta)

    # loading files (for label and mask)
    # with open('../data/generate_materials/label_with_loc_list_new.json') as f:
    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)

    # loc with labels - num and scale
    p_label_num = labels.cpu().detach().numpy().astype(int).sum(0)
    p_label_scale = p_label_num / len(label) * 100

    kfold = KFold(n_splits=fold_num, random_state=seed, shuffle=True)

    train_dict, val_dict = {}, {}

    for alpha in alpha_list:
        fold_flag = 1
        train_d, val_d = {}, {}

        loss_mse = 0
        loc_pred = 0
        loc_logit = 0
        fold_use = 0

        for train_idx, val_idx in kfold.split(label):
            # model = GNN11(g.ndata['feat'].shape[1], 3000, 12, dropout=0.1).to(device)
            # model = GNN12(g.ndata['feat'].shape[1], 300, 100, 12, dropout=0.5).to(device)
            # model = GNN21(g.ndata['feat'].shape[1], 3000, 500, 12, dropout=0.1).to(device)
            # model = GNN22(g.ndata['feat'].shape[1], 400, 200, 80, 12).to(device)
            # model = GNN31(g.ndata['feat'].shape[1], 3000, 1500, 500, 12).to(device)
            model = GNN32(g.ndata['feat'].shape[1], 400, 300, 200, 100, 12).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-6)  # 1e-5 nice

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

                # Compute prediction
                pred = protein_loc_correction(logits, alpha=alpha)

                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                train_loss = multi_loss(logits[train_index], labels[train_index], i_weight)
                train_loss.backward()
                optimizer.step()
                model.eval()
                val_loss = multi_loss(logits[val_index], labels[val_index], i_weight)


                # Compute accuracy on training/validation
                train_aim, train_cov, train_acc, train_atr, train_afr = proformances_record(labels[train_index], pred[train_index])
                val_aim, val_cov, val_acc, val_atr, val_afr = proformances_record(labels[val_index], pred[val_index])

                # record
                # print(type(train_loss.item()), type(val_loss.item())
                #       , type(train_aim), type(train_cov), type(train_acc), type(train_atr), type(train_afr)
                #       , type(val_aim), type(val_cov), type(val_acc), type(val_atr), type(val_afr))
                train_loss_list.append(train_loss.item())  # .clone().detach().cpu().
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
                    print('TIME: {}, In epoch {} / fold {}, learning rate: {:.10f}, alpha: {:.2f}'
                          .format(time, e, fold_flag, lr, alpha))
                    print('tra -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                          .format(train_aim, train_cov, train_acc, train_atr, train_afr, train_loss))
                    print('val -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                          .format(val_aim, val_cov, val_acc, val_atr, val_afr, val_loss))
                    # print(logits)
                    # print(logits.clone().detach().float().cpu().numpy())
                    p_pred = pred.cpu().detach().numpy().astype(int)
                    p_pred_num = p_pred.sum(0)
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
                                .format(lr, fold_num, epoch_num, alpha, device))
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

                # store pred loc mat
                if e == (epoch_num - 1):
                    t = np.array(train_loss_list)
                    v = np.array(val_loss_list)
                    mse = ((t - v) ** 2).sum() / len(t)
                    if mse < loss_mse or loss_mse == 0:
                        loss_mse = mse
                        fold_use = fold_flag
                        loc_pred = pred.clone().detach().int().cpu().numpy()
                        loc_logit = logits.clone().detach().float().cpu().numpy()
            # store record
            train_d[fold_flag] = {
                'aim': train_aim_list, 'cov': train_cov_list, 'acc': train_acc_list,
                'atr': train_atr_list, 'afr': train_afr_list, 'loss': train_loss_list
            }
            val_d[fold_flag] = {
                'aim': val_aim_list, 'cov': val_cov_list, 'acc': val_acc_list,
                'atr': val_atr_list, 'afr': val_afr_list, 'loss': val_loss_list
            }
            fold_flag = fold_flag + 1

        # store loc pred mat
        state = path.split('/')[-2]
        np.save(path + 'loc_pred_' + state + '_f' + str(fold_use) + '_a' + str(alpha), loc_pred)
        np.save(path + 'loc_logit_' + state + '_f' + str(fold_use) + '_a' + str(alpha), loc_logit)

        train_dict[alpha] = train_d
        val_dict[alpha] = val_d

    fig_data = {
        'train': train_dict,
        'validation': val_dict
    }
    with open(path + 'fig_data.json', 'w') as f:
        json.dump(fig_data, f)

    # figures
    dpi = 300
    # classification by alpha
    for alpha in alpha_list:
        # mkdir - alpha
        alpha_path = path + 'a' + str(int(alpha*10)) + '/'
        if not os.path.exists(alpha_path):
            os.mkdir(alpha_path)
        for i in range(fold_num):
            i = i + 1
            # training & validation
            for key in ['aim', 'cov', 'acc', 'atr', 'afr', 'loss']:
                plt.figure(dpi=dpi)
                tra, = plt.plot(epoch, train_dict[alpha][i][key], label='training_1')
                val, = plt.plot(epoch, val_dict[alpha][i][key], label='validation_1')
                plt.xlabel('epoch')
                plt.ylabel(key)
                plt.title(key + ', fold: ' + str(i) + ', alpha: ' + str(alpha))
                plt.legend([tra, val], ['training', 'validation'], loc="best")
                plt.savefig(alpha_path + key + '_fold' + str(i) + '.png')
                plt.close()
    # mix alpha
    for i in range(fold_num):
        for key in ['aim', 'cov', 'acc', 'atr', 'afr', 'loss']:
            plt_num = len(val_dict)
            if plt_num == 1:
                break
            plt.figure(dpi=dpi)
            if plt_num == 2:
                a1, = plt.plot(epoch, val_dict[alpha_list[0]][i + 1][key], label=str(alpha_list[0]))
                a2, = plt.plot(epoch, val_dict[alpha_list[1]][i + 1][key], label=str(alpha_list[1]))
                plt.legend([a1, a2], [str(alpha_list[0]), str(alpha_list[1])], loc="best")
            if plt_num == 3:
                a1, = plt.plot(epoch, val_dict[alpha_list[0]][i + 1][key], label=str(alpha_list[0]))
                a2, = plt.plot(epoch, val_dict[alpha_list[1]][i + 1][key], label=str(alpha_list[1]))
                a3, = plt.plot(epoch, val_dict[alpha_list[2]][i + 1][key], label=str(alpha_list[2]))
                plt.legend([a1, a2, a3], [str(alpha_list[0]), str(alpha_list[1]), str(alpha_list[2])], loc="best")
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.title(key + ', fold: ' + str(i + 1))
            plt.savefig(path + 'alpha_mix_' + key + '_fold' + str(i+1) + '.png')
            plt.close()


