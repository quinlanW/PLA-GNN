"""
Performance
"""
import random
import json
import glob
import numpy as np
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


def performance(file_path="../data/res"):
    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)
    true_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()[label]
    states = ['normal']
    for paths in glob.glob(file_path+'/GSE*'):
        for state in states:
            path = paths + '/' + state + '_logits.npy'
            logit = np.load(path)
            pred = protein_loc_correction(logit, 0.1)[label]
            print(paths.split('/')[-1], state)
            pred_res = performances_record(true_mat, pred)
            print("AIM: {:.3f}, COV: {:.3f}, ACC: {:.3f}".format(pred_res[0], pred_res[1], pred_res[2]))
            random_mat_t = random_pred(pred, True)
            random_mat_f = random_pred(pred, False)
            random_t = performances_record(true_mat, random_mat_t)
            random_f = performances_record(true_mat, random_mat_f)
            print("AIM: {:.3f}, COV: {:.3f}, ACC: {:.3f}".format(random_t[0], random_t[1], random_t[2]))
            print("AIM: {:.3f}, COV: {:.3f}, ACC: {:.3f}".format(random_f[0], random_f[1], random_f[2]))
            print("-" * 20)


if __name__ == '__main__':
    performance()


