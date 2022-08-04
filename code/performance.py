'''
Performance
'''
import random
import json
import numpy as np
from scipy.sparse import load_npz


def random_pred(pred, setnum=True):
    random_mat = np.zeros(shape=pred.shape)
    if setnum:
        pt_num = pred.sum(axis=1)
        for idx in range(len(pt_num)):
            rloc = random.sample(list(range(0, 12)), pt_num[idx])
            random_mat[idx, rloc] = 1
    else:
        for idx in range(len(random_mat)):
            pt_num = random.randint(0, random_mat.shape[1])
            rloc = random.sample(list(range(0, 12)), pt_num)
            random_mat[idx, rloc] = 1
    return random_mat


def proformances_record(loc_true, loc_pred):
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


if __name__ == '__main__':
    pred = np.load('../data/log/normal_b10/loc_pred_normal_b10_f9_a0.1.npy')
    rand_mat_t = random_pred(pred, True)

    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label = json.load(f)
    true = load_npz('../data/generate_materials/loc_matrix.npz').toarray()

    pred = pred[label]
    rand_mat_t = rand_mat_t[label]
    true = true[label]

    print(proformances_record(true, pred))
    print(proformances_record(true, rand_mat_t))

