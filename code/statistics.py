from scipy.sparse import coo_matrix, load_npz
import numpy as np
import copy
import scipy.sparse as sp
import json
import gc

data_dict = {
    1: {1: '../data/generate_materials/GSE30931_data/', 2: 2.75},
    2: {1: '../data/generate_materials/GSE74572_data/', 2: 2.91},
    # 2: {1: '../data/generate_materials/GSE31057_data/', 2: 2.80},  # not used
    # 3: {1: '../data/generate_materials/GSE31057_2_data/', 2: 2.82},  # not used
    4: {1: '../data/generate_materials/GSE27182_data/', 2: 2.99}
}

with open('../data/log/statistics.txt', 'a') as f:
    for val in data_dict.values():
        path = val[1]
        GSE = path.split('/')[-2]
        f.write('#'*20 + ' ' + GSE + ' ' + '#'*20 + '\n')
        ### Network topology adjustment
        ppi_net = load_npz('../data/generate_materials/PPI_normal.npz')  # 正常条件ppi
        pcc_nor = load_npz(path + 'GCN_normal.npz')
        pcc_inter_path = path + 'GCN_inter.npz'
        pcc_inter = load_npz(pcc_inter_path)

        ppi_net = ppi_net.tocsr()
        pcc_normal = pcc_nor.tocsr()
        pcc_interverion = pcc_inter.tocsr()
        diff_matrix = pcc_interverion - pcc_normal  # difference matrix

        # thresholds
        # thrs = [2.00, 2.25, 2.50, 2.75, 3.00]
        # thrs = [2.75, 2.85, 2.95]
        thrs = [val[2]]
        diff_matrix = diff_matrix.toarray()
        diff_std = np.std(diff_matrix)
        diff_mean = np.mean(diff_matrix)
        for thr in thrs:
            ppi_intervention = copy.deepcopy(ppi_net).todense()
            l_threshold = diff_mean - thr * diff_std
            r_threshold = diff_mean + thr * diff_std

            # modify topology
            l_num = (diff_matrix < l_threshold).astype(int).sum()  # 小于左侧阈值的数量
            r_num = (diff_matrix > r_threshold).astype(int).sum()  # 大于右侧阈值的数量
            conn = (ppi_intervention == 1).astype(int).sum()  # 原来存在链接的数量
            res1 = np.logical_and(diff_matrix < l_threshold, ppi_intervention == 1).A  # 原来有连接，且小于左侧阈值 —— 说明需要断开连接
            res_rmv = coo_matrix(res1.astype(int))
            # sp.save_npz(path + str(thr*100) + '-rmv', res_rmv)
            res11 = res1.sum()
            res2 = np.logical_and(diff_matrix > r_threshold, ppi_intervention == 0).A  # 原来没有连接，且大于右侧阈值 —— 说明需要增加连接
            res_add = coo_matrix(res2.astype(int))
            # sp.save_npz(path + str(thr*100) + '-add', res_add)
            res22 = res2.sum()

            print(path)
            print("Threshold: ", thr)
            pcc_all = diff_matrix.shape[0] * diff_matrix.shape[1]
            print("Differential PCC values of protein pairs: ", pcc_all)
            print("Interactions in the control state: ", conn)
            print("Lower than the lower threshold value: ", l_num, "  Percentage: ", l_num/pcc_all*100)
            print("Higher than the upper threshold value: ", r_num, "  Percentage: ", r_num/pcc_all*100)
            f.write(
                "########## Threshold: " + str(thr) + ' ##########\n' +
                "Differential PCC values of protein pairs: " + str(pcc_all) + '\n' +
                "Interactions in the control state: " + str(conn) + '\n' +
                "Lower than the lower threshold value: " + str(l_num) + "  Percentage: " + str(l_num/pcc_all*100) + '\n' +
                "Higher than the upper threshold value: " + str(r_num) + "  Percentage: " + str(r_num / pcc_all * 100) + '\n'
            )

            ppi_intervention[res1] = 0  # 断开连接
            ppi_intervention[res2] = 1  # 增加连接
            alt_conn = (ppi_intervention == 1).astype(int).sum()  # Number of connections after modification  调整拓扑后的连接的数量
            print("number of connection after mod: ", alt_conn)
            print("Removed: ", res11, "  Percentage(before the topology adjustment): ", res11 / conn * 100)
            print("Added: ", res22, "  Percentage(after the topology adjustment): ", res22 / alt_conn * 100)
            print("-----" * 10)
            f.write(
                "Removed: " + str(res11) + "  Percentage(before the topology adjustment): " + str(res11 / conn * 100) + '\n' +
                "Added: " + str(res22) + "  Percentage(after the topology adjustment): " + str(res22 / alt_conn * 100) + '\n'
            )

            # print("=====" * 10)
            ### Co-localization analysis of interacting proteins
            # add = sp.load_npz(path + str(thr*100) + '-add.npz')
            # rmv = sp.load_npz(path + str(thr*100) + '-rmv.npz')

            normal = sp.load_npz('../data/generate_materials/PPI_normal.npz')
            nor_row = list(normal.row)
            nor_col = list(normal.col)
            nor_index = list(zip(nor_row, nor_col))

            ppi_inter_path = path + 'PPI_inter.npz'
            inter = sp.load_npz(ppi_inter_path)
            int_row = list(inter.row)
            int_col = list(inter.col)
            int_index = list(zip(int_row, int_col))

            add = res_add.toarray()
            rmv = res_rmv.toarray()
            loc_mat = sp.load_npz('../data/generate_materials/loc_matrix.npz').toarray()
            loc = loc_mat.sum(1)
            with open('../data/generate_materials/protein_ppi.json') as ff:
                protein = json.load(ff)

            add_index = np.where(add)
            add_index = list(zip(add_index[0], add_index[1]))
            rmv_index = np.where(rmv)
            rmv_index = list(zip(rmv_index[0], rmv_index[1]))

            add_count = {
                'both': 0,
                'single': 0,
                'none': 0,
                'same_add': 0,
                'diff_add': 0,
            }
            for i in add_index:
                protA = i[0]
                protB = i[1]
                locA = loc[protA]
                locB = loc[protB]
                if locA and locB:
                    add_count['both'] += 1
                    lA = loc_mat[protA]
                    lB = loc_mat[protB]
                    res = np.logical_and(lA, lB).any()  # 判断有没有相同的定位
                    if res:
                        add_count['same_add'] += 1
                    else:
                        add_count['diff_add'] += 1
                elif locA or locB:
                    add_count['single'] += 1
                else:
                    add_count['none'] += 1

            rmv_count = {
                'both': 0,
                'single': 0,
                'none': 0,
                'same_rmv': 0,
                'diff_rmv': 0,
            }
            for i in rmv_index:
                protA = i[0]
                protB = i[1]
                locA = loc[protA]
                locB = loc[protB]
                if locA and locB:
                    rmv_count['both'] += 1
                    lA = loc_mat[protA]
                    lB = loc_mat[protB]
                    res = np.logical_and(lA, lB).any()
                    if res:
                        rmv_count['same_rmv'] += 1
                    else:
                        rmv_count['diff_rmv'] += 1
                elif locA or locB:
                    rmv_count['single'] += 1
                else:
                    rmv_count['none'] += 1

            print("Both interacting proteins with annotations & established: ", add_count['both'])
            print("\tInteractions happen within the same subcellular organelle: ", add_count['same_add'])
            print("\tInteractions happen across different subcellular organelles: ", add_count['diff_add'])

            print("Both interacting proteins with annotations & removed: ", rmv_count['both'])
            print("\tInteractions happen within the same subcellular organelle: ", rmv_count['same_rmv'])
            print("\tInteractions happen across different subcellular organelles: ", rmv_count['diff_rmv'])

            f.write(
                "Both interacting proteins with annotations & established: " + str(add_count['both']) + '\n' +
                "\tInteractions happen within the same subcellular organelle: " + str(add_count['same_add']) + '\n' +
                "\tInteractions happen across different subcellular organelles: " + str(add_count['diff_add']) + '\n' +
                "Both interacting proteins with annotations & removed: " + str(rmv_count['both']) + '\n' +
                "\tInteractions happen within the same subcellular organelle: " + str(rmv_count['same_rmv']) + '\n' +
                "\tInteractions happen across different subcellular organelles: " + str(rmv_count['diff_rmv']) + '\n'
            )

            nor_subcellular_count = {
                'same': 0,
                'diff': 0,
                'all': 0
            }
            for i in nor_index:
                protA = i[0]
                protB = i[1]
                locA = loc[protA]
                locB = loc[protB]
                if locA and locB:
                    lA = loc_mat[protA]
                    lB = loc_mat[protB]
                    res = np.logical_and(lA, lB).any()
                    if res:
                        nor_subcellular_count['same'] += 1
                    else:
                        nor_subcellular_count['diff'] += 1

                nor_subcellular_count['all'] = nor_subcellular_count['same'] + nor_subcellular_count['diff']

            print("Interactions have both interacting proteins with annotations: ", nor_subcellular_count['all'])
            print("\tInteractions happen within the same subcellular organelle: ", nor_subcellular_count['same'])
            print("\tInteractions happen across different subcellular organelles: ", nor_subcellular_count['diff'])

            f.write(
                "Interactions have both interacting proteins with annotations: " + str(nor_subcellular_count['all']) + '\n' +
                "\tInteractions happen within the same subcellular organelle: " + str(nor_subcellular_count['same']) + '\n' +
                "\tInteractions happen across different subcellular organelles: " + str(nor_subcellular_count['diff']) + '\n\n'
            )

        f.write('#' * 60 + '\n\n')
        gc.collect()