import json
import torch
import datetime
import torch.nn.functional as F
from torch.nn import MultiLabelSoftMarginLoss
from sklearn.model_selection import KFold


def protein_loc_correction(loc_proba, alpha):
    loc_pred = torch.zeros(loc_proba.shape)
    thresholds = loc_proba.max(1).values * (1 - alpha) + loc_proba.min(1).values * alpha
    for row in range(len(loc_proba)):
        threshold = thresholds[row]
        loc_pred[row][loc_proba[row] > threshold] = 1.
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

        aim = aim + and_set / pred
        cov = cov + and_set / real
        acc = acc + and_set / or_set
        atr = atr + correct
        afr = (or_set - and_set) / len(loc_true[i])

    aim = float(aim / len(loc_true))
    cov = float(cov / len(loc_true))
    acc = float(acc / len(loc_true))
    atr = atr / len(loc_true)
    afr = afr / len(loc_true)

    return aim, cov, acc, atr, afr


def train(g, model, criterion, lr, alpha, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_train_aim = 0
    best_train_cov = 0
    best_train_acc = 0
    best_train_atr = 0
    best_train_afr = 0

    best_val_aim = 0
    best_val_cov = 0
    best_val_acc = 0
    best_val_atr = 0
    best_val_afr = 0

    features = g.ndata['feat']
    labels = g.ndata['loc'].long()

    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)
    kfold = KFold(n_splits=5, random_state=None, shuffle=True)
    fold_flag = 1
    with open('../data/log/performance_log', 'a') as f:
        line = '---------------  alpha: {:.3f}, learning rate: {:.5f}  ---------------\n\r'.format(alpha, lr)
        f.write(line)

    for train_idx, val_idx in kfold.split(label):
        train_index = []
        val_index = []
        for idx in train_idx:
            train_index.append(label[idx])
        for idx in val_idx:
            val_index.append(label[idx])

        for e in range(500):
            # Forward
            logits = model(g, features)  # torch.Size([5000, 12]) <class 'torch.Tensor'>


            # Compute prediction
            pred = protein_loc_correction(logits, alpha=alpha)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = criterion(logits[train_index], labels[train_index])
            # loss = criterion(pred[train_idx], labels[train_idx])  # why this wrong ?

            # Compute accuracy on training/validation
            train_aim, train_cov, train_acc, train_atr, train_afr = proformances_record(labels[train_index], pred[train_index], device)
            val_aim, val_cov, val_acc, val_atr, val_afr = proformances_record(labels[val_index], pred[val_index], device)

            # Save the best training accuracy and the corresponding validation accuracy.
            # train
            if best_train_aim < train_aim:
                best_train_aim = train_aim
            if best_train_cov < train_cov:
                best_train_cov = train_cov
            if best_train_acc < train_acc:
                best_train_acc = train_acc
            if best_train_atr < train_atr:
                best_train_atr = train_atr
            if best_train_afr < train_afr:
                best_train_afr = train_afr
            # val
            if best_val_aim < val_aim:
                best_val_aim = val_aim
            if best_val_cov < val_cov:
                best_val_cov = val_cov
            if best_val_acc < val_acc:
                best_val_acc = val_acc
            if best_val_atr < val_atr:
                best_val_atr = val_atr
            if best_val_afr < val_afr:
                best_val_afr = val_afr

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(logits.is_cuda, pred.is_cuda, loss.is_cuda)
            if e % 5 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('TIME: {}, In epoch {}, loss: {:.8f}, learning rate: {:.5f}, alpha: {:.2f}'.format(time, e, loss, lr, alpha))
                print('training -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}'.format(train_aim, train_cov, train_acc, train_atr, train_afr))
                print('validation -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}'.format(val_aim, val_cov, val_acc, val_atr, val_afr))
                print('-' * 100)
            if e == 499:
                with open('../data/log/performance_log', 'a') as f:
                    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    line = 'TIME: {}, flod {}, loss: {:.8f}\r'.format(time,fold_flag, loss)
                    f.write(line)
                    line = 'Training BEST -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}\r'.format(best_train_aim, best_train_cov, best_train_acc, best_train_atr, best_train_afr)
                    f.write(line)
                    line = 'Validation BEST -- aim: {:.3f}, cov: {:.3f}, acc: {:.3f}, atr: {:.3f}, afr: {:.3f}\n\r'.format(best_val_aim, best_val_cov, best_val_acc, best_val_atr, best_val_afr)
                    f.write(line)
        fold_flag = fold_flag + 1




