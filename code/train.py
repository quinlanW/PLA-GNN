import json
import torch
torch.set_printoptions(profile="full")
import datetime
import matplotlib.pyplot as plt
import numpy as np
from model import *
from sklearn.model_selection import KFold


# def protein_loc_correction(loc_proba, alpha):
#     loc_pred = torch.zeros(loc_proba.shape)
#     thresholds = loc_proba.max(1).values * (1 - alpha) + loc_proba.min(1).values * alpha
#     for row in range(len(loc_proba)):
#         threshold = thresholds[row]
#         loc_pred[row][loc_proba[row] > threshold] = 1.
#     loc_pred = loc_pred.double()
#     return loc_pred


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


def proformances_record(loc_true, loc_pred, device):
    loc_true = loc_true.clone().detach().to(device)
    loc_pred = loc_pred.clone().detach().to(device)
    aim = 0.
    cov = 0.
    acc = 0.
    atr = 0.
    afr = 0.
    for i in range(len(loc_true)):
        and_set = torch.logical_and(loc_true[i], loc_pred[i]).long().sum()
        pred = loc_pred[i].sum().long()
        real = loc_true[i].sum()
        or_set = torch.logical_or(loc_true[i], loc_pred[i]).long().sum()
        correct = 0
        if torch.all(loc_true[i] == loc_pred[i]):
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


def train(g, criterion, lr, alpha, device):
    features = g.ndata['feat']
    labels = g.ndata['loc'].long()

    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)
    kfold = KFold(n_splits=3, random_state=42, shuffle=True)

    fold_flag = 1
    path = '../data/log/'

    for train_idx, val_idx in kfold.split(label):
        model = GCN(g.ndata['feat'].shape[1], 500, 12).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # index conversion
        train_index = []
        val_index = []
        for idx in train_idx:
            train_index.append(label[idx])
        for idx in val_idx:
            val_index.append(label[idx])

        # evaluation indicators
        train_aim_list = []
        train_cov_list = []
        train_acc_list = []
        train_atr_list = []
        train_afr_list = []

        val_aim_list = []
        val_cov_list = []
        val_acc_list = []
        val_atr_list = []
        val_afr_list = []

        epoch = []
        train_loss_list = []
        val_loss_list = []

        for e in range(200):
            # Forward
            logits = model(g, features)  # torch.Size([5000, 12]) <class 'torch.Tensor'>

            # Compute prediction
            pred = protein_loc_correction(logits, alpha=alpha)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            train_loss = criterion(logits[train_index], labels[train_index])
            val_loss = criterion(logits[val_index], labels[val_index])

            # Compute accuracy on training/validation
            train_aim, train_cov, train_acc, train_atr, train_afr = proformances_record(labels[train_index], pred[train_index], device)
            val_aim, val_cov, val_acc, val_atr, val_afr = proformances_record(labels[val_index], pred[val_index], device)

            # record
            epoch.append(e)
            train_loss_list.append(float(train_loss))
            val_loss_list.append(float(val_loss))

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

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if e % 5 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('TIME: {}, In epoch {}, learning rate: {:.5f}, alpha: {:.2f}'.format(time, e, lr, alpha))
                print('tra -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'.format(train_aim, train_cov, train_acc, train_atr, train_afr, train_loss))
                print('val -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}, loss: {:.8f}'.format(val_aim, val_cov, val_acc, val_atr, val_afr, val_loss))

                # print(labels[train_index])
                # print(pred[train_index])

                logits_1 = np.array_str(logits[1].detach().numpy(), max_line_width=np.inf)
                pred_1 = np.array_str(pred[1].detach().numpy().astype(int))
                labels_1 = np.array_str(labels[1].detach().numpy())

                print(logits_1)
                print(pred_1)
                print(labels_1)

                logits_23 = np.array_str(logits[161].detach().numpy(), max_line_width=np.inf)
                pred_23 = np.array_str(pred[161].detach().numpy().astype(int))
                labels_23 = np.array_str(labels[161].detach().numpy())
                print(logits_23)
                print(pred_23)
                print(labels_23)

                logits_100 = np.array_str(logits[5679].detach().numpy(), max_line_width=np.inf)
                pred_100 = np.array_str(pred[5679].detach().numpy().astype(int))
                labels_100 = np.array_str(labels[5679].detach().numpy())
                print(logits_100)
                print(pred_100)
                print(labels_100)

                print('-' * 100)


        # figures
        # loss
        plt.figure(dpi=100)
        tra_loss, = plt.plot(epoch, train_loss_list, label='training')
        val_loss, = plt.plot(epoch, val_loss_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_loss, val_loss], ['training', 'validation'], loc="best")
        plt.savefig(path + 'loss_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # aim
        plt.figure(dpi=100)
        tra_aim, = plt.plot(epoch, train_aim_list, label='training')
        val_aim, = plt.plot(epoch, val_aim_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('aim')
        plt.title('Aim, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_aim, val_aim], ['training', 'validation'], loc="best")
        plt.savefig(path + 'aim_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # cov
        plt.figure(dpi=100)
        tra_cov, = plt.plot(epoch, train_cov_list, label='training')
        val_cov, = plt.plot(epoch, val_cov_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('cov')
        plt.title('Cov, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_cov, val_cov], ['training', 'validation'], loc="best")
        plt.savefig(path + 'cov_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # acc
        plt.figure(dpi=100)
        tra_acc, = plt.plot(epoch, train_acc_list, label='training')
        val_acc, = plt.plot(epoch, val_acc_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Acc, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_acc, val_acc], ['training', 'validation'], loc="best")
        plt.savefig(path + 'acc_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # atr
        plt.figure(dpi=100)
        tra_atr, = plt.plot(epoch, train_atr_list, label='training')
        val_atr, = plt.plot(epoch, val_atr_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('atr')
        plt.title('Atr, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_atr, val_atr], ['training', 'validation'], loc="best")
        plt.savefig(path + 'atr_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # afr
        plt.figure(dpi=100)
        tra_afr, = plt.plot(epoch, train_afr_list, label='training')
        val_afr, = plt.plot(epoch, val_afr_list, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('atr')
        plt.title('Atr, fold: ' + str(fold_flag) + ', alpha: ' + str(alpha))
        plt.legend([tra_afr, val_afr], ['training', 'validation'], loc="best")
        plt.savefig(path + 'afr_fold' + str(fold_flag) + '_alpha' + str(int(alpha*10)) + '.jpg')
        plt.show()

        # text
        # fold = [fold_flag] * len(train_aim_list)
        # epoch = list(range(len(train_aim_list)))
        # fold_log.extend(fold)
        # epoch_log.extend(epoch)
        # train_aim_log.extend(train_aim_list)
        # train_cov_log.extend(train_cov_list)
        # train_acc_log.extend(train_acc_list)
        # train_atr_log.extend(train_atr_list)
        # train_afr_log.extend(train_afr_list)
        # val_aim_log.extend(val_aim_list)
        # val_cov_log.extend(val_cov_list)
        # val_acc_log.extend(val_acc_list)
        # val_atr_log.extend(val_atr_list)
        # val_afr_log.extend(val_afr_list)

        fold_flag = fold_flag + 1

    # fold_log = np.array(fold_log).reshape(len(fold_log), 1)
    # epoch_log = np.array(epoch_log).reshape(len(epoch_log), 1)
    # train_aim_log = np.array(train_aim_log).reshape(len(train_aim_log), 1)
    # val_aim_log = np.array(val_aim_log).reshape(len(val_aim_log), 1)
    # train_cov_log = np.array(train_cov_log).reshape(len(train_cov_log), 1)
    # val_cov_log = np.array(val_cov_log).reshape(len(val_cov_log), 1)
    # train_acc_log = np.array(train_acc_log).reshape(len(train_acc_log), 1)
    # val_acc_log = np.array(val_acc_log).reshape(len(val_acc_log), 1)
    # train_atr_log = np.array(train_atr_log).reshape(len(train_atr_log), 1)
    # val_atr_log = np.array(val_atr_log).reshape(len(val_atr_log), 1)
    # train_afr_log = np.array(train_afr_log).reshape(len(train_afr_log), 1)
    # val_afr_log = np.array(val_afr_log).reshape(len(val_afr_log), 1)
    #
    # data_log = np.concatenate((fold_log, epoch_log, train_aim_log, val_aim_log, train_cov_log, val_cov_log,
    #                        train_acc_log, val_acc_log, train_atr_log, val_atr_log, train_afr_log, val_afr_log), axis=1)
    #
    # np.savetxt(path + 'evaluation_indicators.txt', data_log, fmt="%s", delimiter='  ')  # 需要加入alpha

