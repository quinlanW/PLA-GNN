"""
Performance
"""
import random
import json
import glob
import numpy as np
import os
from scipy.sparse import load_npz


def protein_loc_correction(loc_proba, alpha):  # 24041 12
    """
    regularization

    :param loc_proba: Location probability matrix
    :param alpha: Coefficient
    :return:
        loc_pred - location prediction matrix
    """
    min_proba = loc_proba.min(0)  # 1 12
    max_proba = loc_proba.max(0)  # 1 12
    new_proba = (loc_proba - min_proba) / (max_proba - min_proba)
    sum_proba = new_proba.sum(1).reshape(-1, 1)  # 24041 1
    new_proba = new_proba / sum_proba

    loc_pred = np.zeros(loc_proba.shape)
    thresholds = new_proba.max(1) - (new_proba.max(1) - new_proba.min(1)) * alpha
    for row in range(len(loc_proba)):
        threshold = thresholds[row]
        loc_pred[row][new_proba[row] > threshold] = 1.
    # loc_pred = loc_pred.double()
    return loc_pred


def random_pred(pred, setnum=True):
    """
    Random guess positioning matrix.

    :param pred: Prediction matrix (for determining matrix shape only)
    :param setnum: Set whether the number of guesses localized for each protein is the same as the prediction.
    :return:
        random_mat
    """
    random_mat = np.zeros(shape=pred.shape)
    if setnum:
        pt_num = pred.sum(axis=1).astype(int)
        for idx in range(len(pt_num)):
            rloc = random.sample(list(range(0, 12)), pt_num[idx])
            random_mat[idx, rloc] = 1
    else:
        for idx in range(len(random_mat)):
            pt_num = random.randint(0, random_mat.shape[1])
            rloc = random.sample(list(range(0, 12)), pt_num)
            random_mat[idx, rloc] = 1
    return random_mat


def performances_record(loc_true, loc_pred):
    """
    Performance evaluation

    :param loc_true: Real location matrix.
    :param loc_pred: Predictive location matrix.
    :return:
        performance - [aim, cov, acc]
    """
    mask = np.array([1] * len(loc_true[0]))
    aim = 0.
    cov = 0.
    acc = 0.
    for i in range(len(loc_true)):
        loc_true[i] = np.logical_and(mask, loc_true[i])
        loc_pred[i] = np.logical_and(mask, loc_pred[i])
        and_set = np.logical_and(loc_true[i], loc_pred[i]).sum()
        pred = loc_pred[i].sum()
        real = loc_true[i].sum()
        or_set = np.logical_or(loc_true[i], loc_pred[i]).sum()
        if pred == 0:
            aim = aim + 0
        else:
            aim = aim + and_set / pred
        cov = cov + and_set / real
        acc = acc + and_set / or_set

    aim = float(aim / len(loc_true))
    cov = float(cov / len(loc_true))
    acc = float(acc / len(loc_true))

    return [aim, cov, acc]


def mat_merge(file_path="../data/log"):
    states = ['normal', 'perturbation']
    for paths in glob.glob(file_path+'/GSE*'):
        for state in states:
            for num in range(1, 11):
                log_path = paths + '/' + state
                mat_cnt = np.zeros((24041, 12))

                for path in sorted(glob.glob(log_path + '/' + str(num) + '_*.npy')):
                    mat = np.load(path)
                    mat_cnt += mat
                    roundtime = path.split('/')[-1].split('_')[0]
                mat_cnt /= 10
                np.save(log_path + '/' + state + '_' + str(roundtime) + '_logits.npy', mat_cnt)


def performance(file_path="../data/log"):
    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)
    true_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()[label]
    states = ['normal']
    for paths in glob.glob(file_path+'/GSE*'):
        for state in states:
            print(paths)
            AIMs = []
            COVs = []
            ACCs = []
            for num in range(1, 11):
                path = paths + '/' + state + '/' + state + '_' + str(num) + '_logits.npy'
                logit = np.load(path)
                pred = protein_loc_correction(logit, 0.1)[label]
                pred_res = performances_record(true_mat, pred)
                AIMs.append(pred_res[0])
                COVs.append(pred_res[1])
                ACCs.append(pred_res[2])
            AIM_mean = np.mean(AIMs)
            AIM_std = np.std(AIMs)
            COV_mean = np.mean(COVs)
            COV_std = np.std(COVs)
            ACC_mean = np.mean(ACCs)
            ACC_std = np.std(ACCs)
            print('AIM: {:.3f} +- {:.3f}'.format(AIM_mean, AIM_std))
            print('COV: {:.3f} +- {:.3f}'.format(COV_mean, COV_std))
            print('mlACC: {:.3f} +- {:.3f}'.format(ACC_mean, ACC_std))

    format = np.load('../data/log/GSE74572/normal/normal_1_logits.npy')
    RAIMs_t, RCOVs_t, RmlACCs_t = [], [], []
    RAIMs_f, RCOVs_f, RmlACCs_f = [], [], []

    for i in range(10):
        random_mat_t = random_pred(format, True)
        random_mat_f = random_pred(format, False)
        random_t = performances_record(true_mat, random_mat_t)
        random_f = performances_record(true_mat, random_mat_f)
        RAIMs_t.append(random_t[0])
        RCOVs_t.append(random_t[1])
        RmlACCs_t.append(random_t[2])
        RAIMs_f.append(random_f[0])
        RCOVs_f.append(random_f[1])
        RmlACCs_f.append(random_f[2])
    print('Random limit')
    print('AIM: {:.3f} +- {:.3f}'.format(np.mean(RAIMs_t), np.std(RAIMs_t)))
    print('COV: {:.3f} +- {:.3f}'.format(np.mean(RCOVs_t), np.std(RCOVs_t)))
    print('mlACC: {:.3f} +- {:.3f}'.format(np.mean(RmlACCs_t), np.std(RmlACCs_t)))

    print('Random')
    print('AIM: {:.3f} +- {:.3f}'.format(np.mean(RAIMs_f), np.std(RAIMs_f)))
    print('COV: {:.3f} +- {:.3f}'.format(np.mean(RCOVs_f), np.std(RCOVs_f)))
    print('mlACC: {:.3f} +- {:.3f}'.format(np.mean(RmlACCs_f), np.std(RmlACCs_f)))
    print("-" * 20)


if __name__ == '__main__':
    mat_merge()
    performance()


