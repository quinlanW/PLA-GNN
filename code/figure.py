import numpy as np
import json
import pandas as pd
import csv
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from matplotlib import ticker
from collections import Counter
from scipy.spatial import distance
from scipy.sparse import coo_matrix, load_npz


def save_diff(file_path="../data/generate_materials"):
    '''
    This function stores the difference PCC matrix
    :param file_path:
    :return:
    '''
    ppi = load_npz('../data/generate_materials/PPI_normal.npz').todense().astype(int)
    for paths in tqdm(glob.glob(file_path+'/GSE*')):
        pcc_normal = paths +'/GCN_normal.npz'
        pcc_inter = paths + '/GCN_inter.npz'
        normal = load_npz(pcc_normal)
        inter = load_npz(pcc_inter)
        diff = (inter - normal).todense()
        diff_link = diff[ppi == 1]
        diff_unlink = diff[ppi == 0]
        diff_all = diff.flatten()[0].tolist()
        diff_all.sort()
        pth = paths + '/diff.npy'
        np.save(pth, diff_all)

        linkpth = paths + '/diff_link.npy'
        np.save(linkpth, diff_link)
        unlinkpth = paths + '/diff_unlink.npy'
        np.save(unlinkpth, diff_unlink)


def get_fig_data(path="../data/generate_materials"):
    '''
    This function stores the histogram plot data
    :param path:
    :return:
    '''
    for paths in tqdm(glob.glob(path + '/GSE*')):
        files = ['/diff.npy', '/diff_link.npy', '/diff_unlink.npy']
        hist_data = {
            'all': [],
            'link': [],
            'unlink': []
        }
        for file in files:
            matpath = paths + file
            mat = np.load(matpath).flatten()
            print("load file", matpath)
            print(len(mat))
            pcc_count = [[i, 0] for i in range(0, 201)]
            pcc_bin = []
            c_zero = 0
            for i in range(0, 201):
                pcc_bin.append(-2 + 0.02 * i)
            # print(len(pcc_bin))
            for j in mat:
                # if 0.02 > j > -0.02:
                #     c_zero += 1
                # else:
                pcc_count[int((j - (-2)) / 0.02)][1] += 1
            flag = ''
            if file == '/diff.npy':
                flag = 'all'
            elif file == '/diff_link.npy':
                flag = 'link'
            else:
                flag = 'unlink'
            hist_data[flag].append(pcc_bin)
            hist_data[flag].append(pcc_count)
            hist_path = paths + '/hist_data.json'
            with open(hist_path, 'w') as f:
                json.dump(hist_data, f)


def fig(path="../data/generate_materials"):
    '''
    Histogram plotting
    :param path:
    :return:
    '''
    for paths in tqdm(glob.glob(path + '/GSE*_data')):
        fig_data_path = paths + '/hist_data.json'
        with open(fig_data_path) as f:
            fig_data = json.load(f)
        for item in fig_data:
            bins = fig_data[item][0][0:-1]
            data = list(zip(*fig_data[item][1]))[1][0:-1]
            if item == 'link':
                item = 'interaction'
            if item == 'unlink':
                item = 'without-interaction'
            # plt.title(paths.split('/')[-1].split('_')[0] + ' ' + item)
            plt.rc('font', family='Times New Roman')
            plt.ticklabel_format(style='plain')
            plt.tight_layout()
            plt.yscale("log")
            plt.xlim(-2, 2)
            plt.plot(bins, data)

            # plt.show()
            plt.savefig(path + '/' + paths.split('/')[-1].split('_')[0] + '-' + item, dpi=1200, bbox_inches='tight')
            plt.close()


def subcellular_fig_data():
    '''
    Positioning diversity data statistics (plotting by software plotting, no python code provided)
    :return:
    '''
    loc_data = load_npz('../data/generate_materials/loc_matrix.npz').todense()
    with open('../data/generate_materials/label_list.json') as f:
        label_list = json.load(f)
    with open('../data/generate_materials/label_with_loc_list.json') as f:
        label_list_with_loc = np.array(json.load(f)).astype(int)
    loc_mul_count = [0] * 10
    for item in label_list:
        loc_num = len(item[1])
        loc_mul_count[loc_num] += 1
    print(loc_mul_count)


def fig_alpha():
    '''
    Distribution Charts
    :return:
    '''
    # cal normal
    loc_data = load_npz('../data/generate_materials/loc_matrix.npz').todense()
    all_label_num = int(loc_data.sum())
    normal_loc = loc_data.sum(0)
    normal = normal_loc / all_label_num
    normal = normal.tolist()[0]

    # # print(d)
    # # print(d_data)
    plt.rc('font', family='Times New Roman')

    plt.figure(figsize=(9, 5))
    plt.title('Control')
    d_name = ['Cell cortex', 'Cytosol', 'Actin cytoskeleton', 'Golgi apparatus',
              'Endoplasmic reticulum', 'Nucleolus', 'Peroxisome', 'Mitochondrion',
              'Lysosome', 'Centrosome', 'Nucleus', 'Plasma membrane']
    plt.barh(range(len(normal)), normal, tick_label=d_name)
    plt.xlabel('Percentage')
    plt.tight_layout()
    plt.savefig('../data/normal.png', dpi=1200, bbox_inches='tight')
    plt.close()

    for path in glob.glob('../data/per3*'):
        flag = str(3)
        print(path)
        with open(path) as f:
            f_data = json.load(f)
        for d in f_data:
            d_data = f_data[d]
            d_data = list(map(int, d_data))
            d_data_sum = sum(d_data)
            d_data = (np.array(d_data) / d_data_sum).tolist()
            print(sum(d_data))
            # print(d)
            # print(d_data)
            plt.rc('font', family='Times New Roman')
            dis = distance.jensenshannon(normal, d_data)
            dis = round(dis, 3)
            if d == 'GSE30931':
                dt = 'Bortezomib'
            if d == 'GSE74572':
                dt = 'TSA'
            if d == 'GSE27182':
                dt = 'Tacrolimus'
            plt.figure(figsize=(9, 5))
            plt.title(dt + ', α = 0.' + flag + ' (Jensen-Shannon distance = ' + str(dis) + ')')
            d_name = ['Cell cortex', 'Cytosol', 'Actin cytoskeleton', 'Golgi apparatus',
                      'Endoplasmic reticulum', 'Nucleolus', 'Peroxisome', 'Mitochondrion',
                      'Lysosome', 'Centrosome', 'Nucleus', 'Plasma membrane']
            plt.barh(range(len(d_data)), d_data, tick_label=d_name)
            plt.xlabel('Percentage')
            plt.tight_layout()
            # plt.show()
            plt.savefig('../data/' + dt + 'α=' + flag + '.png', dpi=1200, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    save_diff()
    get_fig_data()
    fig()

    fig_alpha()
    subcellular_fig_data()

