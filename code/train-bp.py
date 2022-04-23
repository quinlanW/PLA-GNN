import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from model import *
from sklearn.model_selection import KFold
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


def multi_loss(input, target, b_weight, i_weight):
    loss = 0
    for i in range(len(b_weight)):
        scl_input = input[:, i]
        scl_target = target[:, i]
        scl_loss = (scl_target * torch.log(scl_input) * i_weight[i] + (1 - scl_target) * torch.log(1 - scl_input)) / (i_weight[i] + 1) * 2
        # scl_loss = scl_target * torch.log(scl_input)
        scl_loss = -scl_loss.sum() / len(input)
        loss += scl_loss * b_weight[i]
    loss = loss / b_weight.sum()

    return loss


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

    # setting loss mode (custom loss / multi label soft margin loss)
    custom_loss = True
    if not custom_loss:
        beta_list = [1]

    for beta in beta_list:
        # mkdir
        beta_path = '../data/log/b' + str(int(beta * 10)) + '/'
        if not os.path.exists(beta_path):
            os.mkdir(beta_path)
        # weight
        loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
        b_weight, i_weight = weight_cal(loc_mat, beta=beta)

        # loss setting
        if not custom_loss:
            criterion = MultiLabelSoftMarginLoss(weight=b_weight)
        else:
            criterion = 0

        # loading files (for label and mask)
        with open('../data/generate_materials/label_with_loc_list.json') as f:
            label = json.load(f)
        with open('../data/generate_materials/label_with_fig.json') as f:
            fig_label = json.load(f)

        # loc with labels - num and scale
        p_label_num = labels.cpu().detach().numpy().astype(int).sum(0)
        p_label_scale = p_label_num / len(label) * 100

        # label mask
        large_label = fig_label[0]
        minor_label = fig_label[1]
        both_label = fig_label[2]

        kfold = KFold(n_splits=fold_num, random_state=42, shuffle=True)

        train_dict, val_dict, large_dict, minor_dict, both_dict = {}, {}, {}, {}, {}
        epoch = list(range(epoch_num))

        for alpha in alpha_list:
            fold_flag = 1
            train_d, val_d, large_d, minor_d, both_d = {}, {}, {}, {}, {}

            for train_idx, val_idx in kfold.split(label):
                # model = GNN2(g.ndata['feat'].shape[1], 3000, 500, 12).to(device)
                model = GNN1(g.ndata['feat'].shape[1], 1000, 12).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # index conversion
                train_index = []
                val_index = []
                for idx in train_idx:
                    train_index.append(label[idx])
                for idx in val_idx:
                    val_index.append(label[idx])

                large_idx = []
                minor_idx = []
                both_idx = []
                for idx in large_label:
                    if idx in val_index:
                        large_idx.append(idx)
                for idx in minor_label:
                    if idx in val_index:
                        minor_idx.append(idx)
                for idx in both_label:
                    if idx in val_index:
                        both_idx.append(idx)

                # evaluation indicators
                train_aim_list, train_cov_list, train_acc_list, train_atr_list, train_afr_list = [], [], [], [], []
                val_aim_list, val_cov_list, val_acc_list, val_atr_list, val_afr_list = [], [], [], [], []
                large_aim_list, large_cov_list, large_acc_list, large_atr_list, large_afr_list = [], [], [], [], []
                minor_aim_list, minor_cov_list, minor_acc_list, minor_atr_list, minor_afr_list = [], [], [], [], []
                both_aim_list, both_cov_list, both_acc_list, both_atr_list, both_afr_list = [], [], [], [], []
                train_loss_list, val_loss_list, large_loss_list, minor_loss_list, both_loss_list = [], [], [], [], []

                for e in range(epoch_num):
                    # Forward
                    logits = model(g, features)
                    # print(type(logits[train_index]), logits[train_index].dtype, logits[train_index].device)
                    # print(type(labels[train_index]), labels[train_index].dtype, labels[train_index].device)

                    # Compute prediction
                    pred = protein_loc_correction(logits, alpha=alpha)

                    # Compute loss
                    # Note that you should only compute the losses of the nodes in the training set.
                    if custom_loss:
                        train_loss = multi_loss(logits[train_index], labels[train_index], b_weight, i_weight)
                        val_loss = multi_loss(logits[val_index], labels[val_index], b_weight, i_weight)

                        large_loss, minor_loss, both_loss = 0, 0, 0
                        if len(large_idx):
                            large_loss = multi_loss(logits[large_idx], labels[large_idx], b_weight, i_weight)
                        if len(minor_idx):
                            minor_loss = multi_loss(logits[minor_idx], labels[minor_idx], b_weight, i_weight)
                        if len(both_idx):
                            both_loss = multi_loss(logits[both_idx], labels[both_idx], b_weight, i_weight)
                    else:
                        train_loss = criterion(logits[train_index], labels[train_index])
                        val_loss = criterion(logits[val_index], labels[val_index])

                        large_loss, minor_loss, both_loss = 0, 0, 0
                        if len(large_idx):
                            large_loss = criterion(logits[large_idx], labels[large_idx])
                        if len(minor_idx):
                            minor_loss = criterion(logits[minor_idx], labels[minor_idx])
                        if len(both_idx):
                            both_loss = criterion(logits[both_idx], labels[both_idx])

                    # Compute accuracy on training/validation
                    train_aim, train_cov, train_acc, train_atr, train_afr = proformances_record(labels[train_index], pred[train_index])
                    val_aim, val_cov, val_acc, val_atr, val_afr = proformances_record(labels[val_index], pred[val_index])
                    #
                    large_aim, large_cov, large_acc, large_atr, large_afr = 0, 0, 0, 0, 0
                    minor_aim, minor_cov, minor_acc, minor_atr, minor_afr = 0, 0, 0, 0, 0
                    both_aim, both_cov, both_acc, both_atr, both_afr = 0, 0, 0, 0, 0
                    if len(large_idx):
                        large_aim, large_cov, large_acc, large_atr, large_afr = proformances_record(labels[large_idx], pred[large_idx])
                    if len(minor_idx):
                        minor_aim, minor_cov, minor_acc, minor_atr, minor_afr = proformances_record(labels[minor_idx], pred[minor_idx])
                    if len(both_idx):
                        both_aim, both_cov, both_acc, both_atr, both_afr = proformances_record(labels[both_idx], pred[both_idx])

                    # record
                    train_loss_list.append(float(train_loss))
                    val_loss_list.append(float(val_loss))
                    if len(large_idx):
                        large_loss_list.append(float(large_loss))
                    if len(minor_idx):
                        minor_loss_list.append(float(minor_loss))
                    if len(both_idx):
                        both_loss_list.append(float(both_loss))

                    train_aim_list.append(train_aim)
                    train_cov_list.append(train_cov)
                    train_acc_list.append(train_acc)
                    train_atr_list.append(train_atr)
                    train_afr_list.append(train_afr)

                    val_aim_list.append(val_aim)
                    val_cov_list.append(val_cov)
                    val_acc_list.append(val_acc)
                    val_atr_list.append(val_atr)
                    val_afr_list.append(val_afr)

                    if len(large_idx):
                        large_aim_list.append(large_aim)
                        large_cov_list.append(large_cov)
                        large_acc_list.append(large_acc)
                        large_atr_list.append(large_atr)
                        large_afr_list.append(large_afr)
                    if len(minor_idx):
                        minor_aim_list.append(minor_aim)
                        minor_cov_list.append(minor_cov)
                        minor_acc_list.append(minor_acc)
                        minor_atr_list.append(minor_atr)
                        minor_afr_list.append(minor_afr)
                    if len(both_idx):
                        both_aim_list.append(both_aim)
                        both_cov_list.append(both_cov)
                        both_acc_list.append(both_acc)
                        both_atr_list.append(both_atr)
                        both_afr_list.append(both_afr)

                    # Backward
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    if e % 5 == 0 or e == epoch_num:
                        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print('TIME: {}, In epoch {} /fold {}, learning rate: {:.10f}, alpha: {:.2f}, beta: {:.2f}'
                              .format(time, e, fold_flag, lr, alpha, beta))
                        print('tra -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                              .format(train_aim, train_cov, train_acc, train_atr, train_afr, train_loss))
                        print('val -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'
                              .format(val_aim, val_cov, val_acc, val_atr, val_afr, val_loss))
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
                        txt_path = beta_path + 'txt_log.txt'
                        with open(txt_path, 'a') as f:
                            if e == 0:
                                f.write('-' * 190)
                                f.write('\n')
                                f.write(
                                    'learning rate:{:.8f}, fold num:{}, epoch num:{}, alpha:{}, beta:{}, device:{}\n'
                                    .format(lr, fold_num, epoch_num, alpha, beta, device))
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

                train_d[fold_flag] = {
                    'aim': train_aim_list, 'cov': train_cov_list, 'acc': train_acc_list,
                    'atr': train_atr_list, 'afr': train_afr_list, 'loss': train_loss_list
                }
                val_d[fold_flag] = {
                    'aim': val_aim_list, 'cov': val_cov_list, 'acc': val_acc_list,
                    'atr': val_atr_list, 'afr': val_afr_list, 'loss': val_loss_list
                }
                if len(large_idx):
                    large_d[fold_flag] = {
                        'aim': large_aim_list, 'cov': large_cov_list, 'acc': large_acc_list,
                        'atr': large_atr_list, 'afr': large_afr_list, 'loss': large_loss_list
                    }
                if len(minor_idx):
                    minor_d[fold_flag] = {
                        'aim': minor_aim_list, 'cov': minor_cov_list, 'acc': minor_acc_list,
                        'atr': minor_atr_list, 'afr': minor_afr_list, 'loss': minor_loss_list
                    }
                if len(both_idx):
                    both_d[fold_flag] = {
                        'aim': both_aim_list, 'cov': both_cov_list, 'acc': both_acc_list,
                        'atr': both_atr_list, 'afr': both_afr_list, 'loss': both_loss_list
                    }
                fold_flag = fold_flag + 1

            train_dict[alpha] = train_d
            val_dict[alpha] = val_d
            large_dict[alpha] = large_d
            minor_dict[alpha] = minor_d
            both_dict[alpha] = both_d

        # store
        # data_dict = {
        #     'train': train_dict,
        #     'val': val_dict,
        #     'large': large_dict,
        #     'minor': minor_dict,
        #     'both': both_dict
        # }
        # with open('../data/log/data.json', 'w') as f:
        #     json.dump(data_dict, f)
        # figures
        dpi = 100
        for alpha in alpha_list:
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
                    plt.savefig(beta_path + key + '_a' + str(alpha) + 'f' + str(i) + '.png')
                    plt.close()
                # large & minor & both
                for key in ['aim', 'cov', 'acc', 'atr', 'afr', 'loss']:
                    plt.figure(dpi=dpi)
                    val, = plt.plot(epoch, val_dict[alpha][i][key], label='validation')
                    large = plt.plot()
                    minor = plt.plot()
                    both = plt.plot()
                    if large_dict[alpha][i]:
                        large, = plt.plot(epoch, large_dict[alpha][i][key], label='large')
                    if minor_dict[alpha][i]:
                        minor, = plt.plot(epoch, minor_dict[alpha][i][key], label='minor')
                    if both_dict[alpha][i]:
                        both, = plt.plot(epoch, both_dict[alpha][i][key], label='both')
                    plt.xlabel('epoch')
                    plt.ylabel(key)
                    plt.title(key + ', fold: ' + str(i) + ', alpha: ' + str(alpha))
                    plt.legend([val, large, minor, both], ['validation', 'large', 'minor', 'both'], loc="best")
                    plt.savefig(beta_path + 'mix_' + key + '_a' + str(alpha) + 'f' + str(i) + '.png')
                    plt.close()
                # alpha & validation
                # for key in ['aim', 'cov', 'acc', 'atr', 'afr', 'loss']:
                #     plt.figure(dpi=dpi)
                #     val_1, = plt.plot(epoch, val_dict[alpha_list[0]][i][key], label='validation_1')
                #     val_2, = plt.plot(epoch, val_dict[alpha_list[1]][i][key], label='validation_2')
                #     val_3, = plt.plot(epoch, val_dict[alpha_list[2]][i][key], label='validation_3')
                #     plt.xlabel('epoch')
                #     plt.ylabel(key)
                #     plt.title(key + ', fold: ' + str(i))
                #     plt.legend([val_1, val_2, val_3], ['aplha-0.1', 'aplha-0.2', 'aplha-0.3'], loc="best")
                #     plt.savefig(beta_path + key + '_fold-' + str(i) + '.png')
                #     # plt.show()
                #     plt.close()


