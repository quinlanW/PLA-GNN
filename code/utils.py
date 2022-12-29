'''
Utils
'''
import dgl
import json
import torch as th
import numpy as np
import matplotlib.pyplot as plt


def data_normalize(mat):
    mat_normalize = mat.copy()
    p_mean = mat_normalize.mean(0)
    p_std = mat_normalize.std(0)
    for j in range(mat_normalize.shape[1]):
        mat_normalize[:, j] = (mat_normalize[:, j] - p_mean[j]) / p_std[j]

    return mat_normalize


def create_graph(ppi, ecc, gcn, loc, expr, uniprot):
    start = list(ppi.row)
    end = list(ppi.col)
    num_nodes = len(uniprot)
    g = dgl.graph((start, end), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.nodes[list(range(num_nodes))].data['loc'] = th.from_numpy(loc.toarray().astype(np.float32))
    edge_feat = np.hstack((gcn, ecc))
    node_feat = th.tensor(np.hstack((expr, edge_feat)), dtype=th.float)
    g.nodes[list(range(num_nodes))].data['feat'] = node_feat

    return g


def fig_and_perf(path):
    with open(path) as f:
        fig_data = json.load(f)

    val_data = fig_data['validation']
    length = len(val_data['0.1']['1']['aim'])
    f_num = len(val_data['0.1'])
    f_data = {
        'AIM': {},
        'COV': {},
        'mlACC': {}
    }

    GSE = path.split('/')[3] + '-' + path.split('/')[4]
    for alpha in val_data:
        aim, cov, acc = np.array([0.0]*length), np.array([0.0]*length), np.array([0.0]*length)
        for fold in val_data[alpha].values():
            aim += np.array(fold['aim'])
            cov += np.array(fold['cov'])
            acc += np.array(fold['acc'])
        aim /= f_num
        cov /= f_num
        acc /= f_num
        f_data['AIM'][alpha] = aim
        f_data['COV'][alpha] = cov
        f_data['mlACC'][alpha] = acc

    for item in f_data:
        plt.figure(dpi=600)
        for alpha in f_data[item]:
            plt.plot(list(range(length)), f_data[item][alpha], label=alpha)
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel(item)
        plt.title(GSE)
        plt.show()


if __name__ == '__main__':
    # fig_and_perf('../data/log/GSE27182/normal/fig_data.json')
    # fig_and_perf('../data/log/GSE27182/perturbation/fig_data.json')
    # fig_and_perf('../data/log/GSE74572/normal/fig_data.json')
    # fig_and_perf('../data/log/GSE74572/perturbation/fig_data.json')
    # figures
    dpi = 300
    # classification by alpha
    with open('../data/fig_data.json') as f:
        fig_data = json.load(f)

    # for i in fig_data['train']['0.1']:
    #     # training & validation
    #     for key in ['aim', 'cov', 'acc', 'loss']:
    #         plt.figure(dpi=600)
    #         tra, = plt.plot(list(range(300)), fig_data['train']['0.1'][i][key], label='training')
    #         val, = plt.plot(list(range(300)), fig_data['validation']['0.1'][i][key], label='validation')
    #         plt.xlabel('epoch')
    #         plt.ylabel(key)
    #         plt.title(key + i)
    #         plt.legend([tra, val], ['training', 'validation'], loc="best")
    #         # plt.savefig(alpha_path + key + '_fold' + str(i) + '.png')
    #         #
    #         plt.show()
    #         plt.close()

    tra_data = fig_data['train']
    val_data = fig_data['validation']
    length = len(val_data['0.1']['1']['aim'])
    f_num = len(val_data['0.1'])
    f_data = {
        'AIM': {
            'tra': [],
            'val': []
        },
        'COV': {
            'tra': [],
            'val': []
        },
        'mlACC': {
            'tra': [],
            'val': []
        },
        'loss': {
            'tra': [],
            'val': []
        }
    }
    for data in fig_data:
        for item in fig_data[data].values():
            aim, cov, acc, loss = np.array([0.0] * length), np.array([0.0] * length), np.array([0.0] * length), np.array([0.0] * length)
            for fold in item.values():
                aim += np.array(fold['aim'])
                cov += np.array(fold['cov'])
                acc += np.array(fold['acc'])
                loss += np.array(fold['loss'])
            aim /= f_num
            cov /= f_num
            acc /= f_num
            loss /= f_num
            if data == 'train':
                f_data['AIM']['tra'] += list(aim)
                f_data['COV']['tra'] += list(cov)
                f_data['mlACC']['tra'] += list(acc)
                f_data['loss']['tra'] += list(loss)
            elif data == 'validation':
                f_data['AIM']['val'] += list(aim)
                f_data['COV']['val'] += list(cov)
                f_data['mlACC']['val'] += list(acc)
                f_data['loss']['val'] += list(loss)

    for item in f_data:
        plt.figure(dpi=600)
        plt.plot(list(range(length)), f_data[item]['tra'], label='tra')
        plt.plot(list(range(length)), f_data[item]['val'], label='val')
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel(item)
        plt.show()
        plt.close()