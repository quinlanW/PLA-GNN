import os
import json
import csv
import gzip
import urllib.parse
import urllib.request
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from pathlib import Path
from tqdm import tqdm


def extract_interaction_data(data_file):
    biogrid_id_list = set()  # store biogrid id or the protein in the network <str>
    interaction_list = set()  # store protein interactions in the network <tuple>

    with open(data_file) as f:
        next(f)
        biogrid_data = f.readlines()

    for line in tqdm(biogrid_data, desc='Extracting interaction information'):  # for line in biogrid_data:
        line = line.split('\t')
        if '0915' in line[11] or '0407' in line[11]:  # select interactionType
            id_1 = line[2].split('|')[0].split(':')[1]
            id_2 = line[3].split('|')[0].split(':')[1]
            # The format of line[2] and line[3]: biogrid:100010|...
            if id_1 == id_2:  # exclusion of identical protein interactions
                continue
            biogrid_id_list.add(id_1)
            biogrid_id_list.add(id_2)
            interaction_list.add((id_1, id_2))
            interaction_list.add((id_2, id_1))

    biogrid_id_list = list(biogrid_id_list)  # dividing the same elements
    biogrid_id_list.sort()
    interaction_list = list(interaction_list)

    return_dict = {
        'id_list': biogrid_id_list,
        'interaction_list': interaction_list
    }

    return return_dict


def id_mapping(query_list, original_id, target_id, direction=0, series=''):
    with tqdm(total=2, desc='ID Mapping') as idmap_bar:
        query_str = ''
        for item in query_list:
            item = item.strip() + ' '
            query_str += item

        url = 'https://www.uniprot.org/uploadlists/'
        params = {
            'from': original_id,
            'to': target_id,
            'format': 'tab',
            'query': query_str
        }

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read().decode('utf-8')
        idmap_bar.update()

        mapping_dict = {}
        for item in response.strip().split('\n')[1:]:
            orig_id = item.split()[0]
            targ_id = item.split()[1]
            if direction == 0:
                if orig_id in mapping_dict:
                    mapping_dict[orig_id].append(targ_id)
                else:
                    mapping_dict[orig_id] = [targ_id, ]
            else:
                if targ_id in mapping_dict:
                    mapping_dict[targ_id].append(orig_id)
                else:
                    mapping_dict[targ_id] = [orig_id, ]
        idmap_bar.update()
    idmap_bar.close()

    return mapping_dict


def construct_uniprot_ppi(mapping_dict, interaction_list):
    uniprot_list = []  # The set of nodes in the interaction network
    uniprot_interaction_list = []  # The interaction protein list
    for item in tqdm(interaction_list, desc='Protein interaction ID mapping'):
        if item[0] in mapping_dict and item[1] in mapping_dict:
            for uni_1 in mapping_dict[item[0]]:
                for uni_2 in mapping_dict[item[1]]:
                    uniprot_interaction_list.append((uni_1, uni_2))
                    uniprot_list.append(uni_1)
                    uniprot_list.append(uni_2)

    uniprot_list = list(set(uniprot_list))
    uniprot_list.sort()

    node_set = set()
    for interaction in tqdm(uniprot_interaction_list, desc='record coordinates'):
        uni_1 = uniprot_list.index(interaction[0])  # row
        uni_2 = uniprot_list.index(interaction[1])  # col
        node_set.add((uni_1, uni_2))
        node_set.add((uni_2, uni_1))

    ppi_row = []
    ppi_col = []
    ppi_data = []
    for node in tqdm(node_set, desc='constructing PPI adjacency matrix'):
        x = node[0]
        y = node[1]
        ppi_row.append(x)
        ppi_col.append(y)
        ppi_data.append(1)

    ppi = coo_matrix((ppi_data, (ppi_row, ppi_col)), shape=(len(uniprot_list), len(uniprot_list)))
    ppi.setdiag(0)
    ppi.eliminate_zeros()

    return ppi, uniprot_list


def matrix_deflation(mat, uni_list, start, end):
    # (23867, 23867)
    if start != 0 and end != 0:
        mat = mat.toarray()[start:end, start:end]
        uni_list = uni_list[start: end]
        mat = coo_matrix(mat)

    return mat, uni_list


def construct_normal_ppi(data='../data/support_materials/BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt', start=0, end=0):
    interaction_data = extract_interaction_data(data_file=data)  # 蛋白质相互作用json文件
    map_dict = id_mapping(query_list=interaction_data['id_list'], original_id='BIOGRID_ID', target_id='ACC', direction=0, series='')  # 映射json文件
    ppi, protein_list = construct_uniprot_ppi(mapping_dict=map_dict, interaction_list=interaction_data['interaction_list'])  # 蛋白质相互作用网络npz文件，相互作用网络中蛋白质列表json文件
    mat, uniprot_list = matrix_deflation(ppi, protein_list, start, end)

    return mat, uniprot_list


def extract_expression_data_from_soft_file(data_file):
    with open(data_file) as f:
        file_data = f.readlines()

    series = ''
    probe_gene_dict = {}  # store probe 2 entrez gene
    probe_expr_dict = {}  # store expression matrix
    entrez_gene_id_list = []  # store platform probe correspond enterz gene id

    not_data_flag = ['!platform_table_begin', '!sample_table_begin', 'ID_REF	VALUE']
    col_name_flag = 'ENTREZ_GENE_ID'
    sample_id_flag = False
    platform_append_flag = False
    matrix_append_flag = False
    probe_ref = 0
    entrez_gene_ref = 0
    for line in tqdm(file_data, desc='Extracting expression information'):  # for line in file_data:
        # for platform data
        if line.startswith('^SERIES'):
            series = line.split()[-1].strip()  # get series
        if line.startswith('!platform_table_begin'):
            platform_append_flag = True
        if line.startswith('!platform_table_end'):
            platform_append_flag = False
        if platform_append_flag:  # get entrez gene id and correspond probe id
            if col_name_flag in line.strip().split('\t'):
                probe_ref = line.strip().split('\t').index('ID')
                entrez_gene_ref = line.strip().split('\t').index('ENTREZ_GENE_ID')
            elif line.strip() not in not_data_flag:
                # for mapping dict
                probe = line.split('\t')[probe_ref].strip()
                gene_id_str = line.split('\t')[entrez_gene_ref].strip()
                if gene_id_str:  # 存在大量的probe没有对应的entrez gene id ！！！！！！
                    for gene_id in gene_id_str.split('///'):
                        gene_id = gene_id.strip()
                        if probe in probe_gene_dict:
                            probe_gene_dict[probe].append(gene_id)
                        else:
                            probe_gene_dict[probe] = [gene_id, ]  # first time
        # for expression matrix data
        sample_id = ''
        if line.startswith('^SAMPLE'):
            sample_id_flag = True
            sample_id = line.split('=')[1].strip()
            if 'probe_id' in probe_expr_dict:
                probe_expr_dict['probe_id'].append(sample_id)
            else:
                probe_expr_dict['probe_id'] = [sample_id, ]  # first time
        if sample_id_flag:
            if line.startswith('!sample_table_begin'):
                matrix_append_flag = True
            if line.startswith('!sample_table_end'):
                matrix_append_flag = False
                sample_id_flag = False
            if matrix_append_flag:
                if line.strip() not in not_data_flag:
                    probe = line.split()[0].strip()
                    value = float(line.split()[1].strip())
                    if probe in probe_expr_dict:
                        probe_expr_dict[probe].append(value)
                    else:
                        probe_expr_dict[probe] = [value, ]  # first time

    gene_expr_dict = {}
    gene_expr_dict['id'] = probe_expr_dict['probe_id']
    for probe, genes in tqdm(probe_gene_dict.items(), desc='construct gene expression dict'):
        for gene in genes:
            if gene in gene_expr_dict:
                gene_expr_dict[gene].append(probe_expr_dict[probe])
            else:
                gene_expr_dict[gene] = [probe_expr_dict[probe], ]
            # entrez gene id list
            entrez_gene_id_list.append(gene)

    return_dict = {
        'series': series,
        'id_list': entrez_gene_id_list,
        'expr_dict': gene_expr_dict
    }

    data_path = '../data/generate_materials/' + series + '_data'
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)

    return return_dict


def extract_expression_data_from_csv_file(data_file):
    """
    Extracting expression data from CSV format files

    :param data_file: File path
    :param file: Generate files
    :return: Dictionary with series, ids and expression data
    """
    series = data_file.split('/')[-1].split('_')[0]
    ensemble_id_list = []
    expr_dict = {}
    head_flag = True
    with open(data_file) as csv_file:
        data = csv.reader(csv_file)
        for line in tqdm(data, desc='Extracting expression information'):  # for line in data:
            ensemble_id = line[0]
            values = line[1:]
            if head_flag:
                head_flag = False
                expr_dict['id'] = values
                continue
            values = list(map(float, values))
            expr_dict[ensemble_id] = values
            ensemble_id_list.append(ensemble_id)

    return_dict = {
        'series': series,
        'id_list': ensemble_id_list,
        'expr_dict': expr_dict
    }

    data_path = '../data/generate_materials/' + series + '_data'
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)

    return return_dict


def construct_protein_expression_dict(expression_data, map_dict, uniprot_list, series):
    protein_expr_dict = {'id': expression_data['id']}
    for protein_id in tqdm(expression_data, desc='construct protein expression dict'):
        if protein_id in map_dict:  # Determine if the ID is in the mapping
            protein_list = map_dict[protein_id]  # Get the list of proteins corresponding to the ID
            for protein in protein_list:
                if protein in uniprot_list:  # Determine if a protein is in the set of interaction network nodes
                    if protein not in protein_expr_dict:
                        # protein_expr_dict[protein] = [expression_data[protein_id], ]
                        protein_expr_dict[protein] = expression_data[protein_id]
                    else:
                        for item in expression_data[protein_id]:
                            protein_expr_dict[protein].append(item)

    return protein_expr_dict


def construct_co_expression_matrix(sample_dict, uniprot_list, expression_data, series, method='median'):
    ncol = len(expression_data['id'])
    nrow = len(uniprot_list)

    expr = np.zeros((nrow, ncol))  # 表达矩阵 -- 根据method将一个protein对应的多行数据处理为一行
    co_expr = np.zeros((nrow, nrow))
    method = np.median if method == 'median' else eval('np.ndarray.' + method)

    # 处理蛋白质表达矩阵
    for protein in tqdm(uniprot_list, desc='processing protein expression matrix'):
    # for protein in uniprot_list:
        if protein in expression_data:
            protein_ref = uniprot_list.index(protein)  # 得到对应蛋白质的下标
            values = np.array(expression_data[protein])
            # print(protein, values.shape)
            expr[protein_ref] = method(values, 0)  # 通过method处理成一行

    normal_ref = []  # 正常样本下标
    intervention_ref = []  # 干预样本下标
    # 分离正常样本和药物干预样本
    for flag, samples in tqdm(sample_dict.items(), desc='separate samples'):
        # for flag, samples in sample_dict.items():
        if flag == 'normal':
            for sample in samples:
                normal_ref.append(expression_data['id'].index(sample))
        elif flag == 'intervention':
            for sample in samples:
                intervention_ref.append(expression_data['id'].index(sample))

    normal_expr = np.zeros((nrow, 0))
    intervention_expr = np.zeros((nrow, 0))

    # 构建不同条件下的表达矩阵
    for ref in tqdm(normal_ref, desc='normal samples expression matrix construction'):
        # for ref in normal_ref:
        sample_expr = expr[0:, ref: ref + 1]
        normal_expr = np.hstack((normal_expr, sample_expr))
    for ref in tqdm(intervention_ref, desc='intervened expression matrix construction'):
        # for ref in intervetion_ref:
        sample_expr = expr[0:, ref: ref + 1]
        intervention_expr = np.hstack((intervention_expr, sample_expr))

    with tqdm(total=5, desc='co-expression correlation coefficient matrix construction') as coexpr_bar:
        # 计算共表达相关系数（皮尔逊相关系数）
        pcc_normal = np.corrcoef(normal_expr)
        coexpr_bar.update()
        pcc_intervention = np.corrcoef(intervention_expr)
        coexpr_bar.update()
        np.fill_diagonal(pcc_normal, 0)  # 填充对角线
        np.fill_diagonal(pcc_intervention, 0)
        coexpr_bar.update()

        # 转化为稀疏矩阵
        pcc_normal_nan = np.isnan(pcc_normal)  # nan元素位置 True
        pcc_intervetion_nan = np.isnan(pcc_intervention)
        pcc_normal[pcc_normal_nan] = 0  # nan替换为0 不替换的话稀疏矩阵会存nan
        pcc_intervention[pcc_intervetion_nan] = 0
        pcc_nor = coo_matrix(pcc_normal)
        coexpr_bar.update()
        pcc_inter = coo_matrix(pcc_intervention) # 考虑csr
        coexpr_bar.update()
    coexpr_bar.close()

    return pcc_nor, pcc_inter


def construct_disease_comparison_gcn(data, uniprot_list, sample, file_type='soft'):
    expr_data, map_dict = {}, {}
    if file_type == 'soft':
        expr_data = extract_expression_data_from_soft_file(data_file=data)  # 基因表达json文件
        map_dict = id_mapping(query_list=expr_data['id_list'], original_id='P_ENTREZGENEID', target_id='ACC', direction=0, series=expr_data['series'])  # 映射json文件
    elif file_type == 'csv':
        expr_data = extract_expression_data_from_csv_file(data_file=data)  # 基因表达json文件
        map_dict = id_mapping(query_list=expr_data['id_list'], original_id='ENSEMBL_ID', target_id='ACC', direction=0, series=expr_data['series'])  # 映射json文件

    protein_expr = construct_protein_expression_dict(expression_data=expr_data['expr_dict'], map_dict=map_dict, uniprot_list=uniprot_list, series=expr_data['series'])  # 蛋白质表达json文件
    gcn_normal, gcn_disease = construct_co_expression_matrix(sample_dict=sample, uniprot_list=uniprot_list, expression_data=protein_expr, series=expr_data['series'], method='median')  # 两种条件下的共表达相关系数网络npz文件

    return gcn_normal, gcn_disease, expr_data['series']


def edge_clustering_coefficients(ppi_net, epsilon=0):
    ppi = ppi_net.tocsr()
    ecc_row = []
    ecc_col = []
    ecc_data = []
    for i in tqdm(range(ppi.shape[0]), desc='construct ecc matrix'):
        i_data = ppi[i].toarray()
        neighbors = ppi[i].indices
        degree_i = ppi[i].data.sum()
        for j in neighbors[neighbors > i]:
            j_data = ppi[j].toarray()
            triangles_num = np.logical_and(i_data, j_data).sum()
            degree_j = ppi[j].data.sum()
            possibly_triangles_num = min(degree_i, degree_j) - 1
            if possibly_triangles_num == 0:
                value = epsilon
            else:
                value = triangles_num / possibly_triangles_num

            ecc_row.append(i)
            ecc_col.append(j)
            ecc_data.append(value)
            ecc_row.append(j)
            ecc_col.append(i)
            ecc_data.append(value)

    ecc = coo_matrix((ecc_data, (ecc_row, ecc_col)), shape=ppi.shape)

    return ecc


def modify_network_topology(ppi_net, pcc_nor, pcc_inter, series):
    with tqdm(total=5, desc='modify protein interaction network') as mod_bar:
        ppi = ppi_net.tocsr()
        pcc_normal = pcc_nor.tocsr()
        pcc_interverion = pcc_inter.tocsr()
        mod_bar.update()

        diff_matrix = pcc_interverion - pcc_normal  # difference matrix
        ppi_intervention = ppi.copy()
        mod_bar.update()

        # connected
        conn_diff = diff_matrix.copy()
        conn_diff[ppi == 0] = 0  # remove values without interactions
        conn_std = diff_matrix[ppi == 1].std()
        conn_mean = diff_matrix[ppi == 1].mean()
        conn_l_threshold = conn_mean - 5 * conn_std
        conn_r_threshold = conn_mean + 5 * conn_std
        mod_bar.update()
        # unconnected
        unconn_diff = diff_matrix.copy()
        unconn_diff[ppi == 1] = 0  # removing values with interactions
        unconn_std = diff_matrix[ppi == 0].std()
        unconn_mean = diff_matrix[ppi == 0].mean()
        unconn_l_threshold = unconn_mean - 5 * unconn_std
        unconn_r_threshold = unconn_mean + 5 * unconn_std
        mod_bar.update()

        # modify topology
        ppi_intervention[conn_diff < conn_l_threshold] = 0
        ppi_intervention[conn_diff > conn_r_threshold] = 0
        ppi_intervention[unconn_diff < unconn_l_threshold] = 1
        ppi_intervention[unconn_diff > unconn_r_threshold] = 1
        # ppi_inter = coo_matrix(ppi_intervention)
        mod_bar.update()
    mod_bar.close()

    return ppi_intervention


def construct_matrix_of_normal_and_disease_cond(data, start, end):
    ppi_normal, protein_list = construct_normal_ppi(start=start, end=end)
    ecc_normal = edge_clustering_coefficients(ppi_net=ppi_normal)  # 正常条件ecc npz文件

    for sample_data in data.values():
        file_type = sample_data[1].split('.')[-1]
        gcn_normal, gcn_disease, series = construct_disease_comparison_gcn(data=sample_data[1],
                                                                           uniprot_list=protein_list,
                                                                           sample=sample_data[2], file_type=file_type)
        ppi_disease = modify_network_topology(ppi_net=ppi_normal, pcc_nor=gcn_normal, pcc_inter=gcn_disease,
                                              series=series)  # 疾病条件ppi npz文件
        ecc_disease = edge_clustering_coefficients(ppi_net=ppi_disease)  # 疾病状态ecc npz文件

        normal_path = '../data/generate_materials/'
        protein_list_path = normal_path + 'protein_ppi.json'  # 正常条件下蛋白质相互作用网络中的的蛋白质节点
        if not os.path.exists(normal_path + 'PPI_normal.npz'):
            sparse.save_npz(normal_path + 'PPI_normal', ppi_normal)
        if not os.path.exists(normal_path + 'ECC_normal.npz'):
            sparse.save_npz(normal_path + 'ECC_normal', ecc_normal)
        if not os.path.exists(normal_path + 'GCN_normal'):
            sparse.save_npz(normal_path + 'GCN_normal', gcn_normal)
        if not Path(protein_list_path).exists():
            with open(protein_list_path, 'w') as f:
                json.dump(protein_list, f)

        disease_path = '../data/generate_materials/' + series + '_data/'
        if not os.path.exists(disease_path + 'PPI_disease.npz'):
            sparse.save_npz(disease_path + 'PPI_disease', ppi_disease)
        if not os.path.exists(disease_path + 'ECC_disease.npz'):
            sparse.save_npz(disease_path + 'ECC_disease', ecc_disease)
        if not os.path.exists(disease_path + 'GCN_disease.npz'):
            sparse.save_npz(disease_path + 'GCN_disease', gcn_disease)


def judge_gene_onthology_line(line, go_list):
    if line.startswith('DR   GO;') and 'C:' in line and ('IDA' in line or 'HDA' in line) and line[9:19] in go_list:
        return True
    else:
        return False


def extract_localization_data(uniprot_sprot_data='../data/support_materials/uniprot_sprot_human.dat.gz'):
    with tqdm(total=3, desc='cellular component data reading') as loc_bar:
        with gzip.open(uniprot_sprot_data) as f:
            data = f.read().decode()
        entry_list = data.split('//\n')[0: -1]  # split each protein data
        loc_dict = {}  # protein and corresponding CC
        loc_bar.update()

        with open('../data/support_materials/cellular_component.txt') as f:
            loc_list = f.read().split()
        loc_bar.update()

        with open('../data/generate_materials/protein_ppi.json') as f:
            uni_list = json.load(f)
        loc_bar.update()
    loc_bar.close()

    for entry in tqdm(entry_list, desc='cellular component data extracting'):
        AC, CC_list = None, []
        lines = entry.split('\n')
        for line in lines:
            if not AC:
                if line.startswith('AC'):
                    AC = line.split()[1].replace(';', '')
            elif judge_gene_onthology_line(line, loc_list):
                CC = line[9: 19]
                CC_list.append(CC)
        if AC in uni_list and CC_list:
            loc_dict[AC] = CC_list

    label_list = []
    for item in uni_list:
        if item in loc_dict.keys():
            loc = loc_dict[item]
        else:
            loc = []
        label_item = (item, loc)
        label_list.append(label_item)

    return label_list


def generate_virtual_protein(label_list):
    vir_label_list = []
    for protein, locs in label_list:
        if len(locs) > 1:
            for flag in range(len(locs)):
                vir_protein = protein + '_' + str(flag)
                loc = locs[flag]
                vir_label_list.append((vir_protein, [loc, ]))
        else:
            vir_label_list.append((protein, locs))

    return vir_label_list


def construct_protein_loc_matrix(label_list):
    """
    Construction of protein localization annotation matrix

    :param vir_loc_list: Virtual positioning protein list.
    :param file: Generate files
    :return: protein localization annotation matrix
    """
    with open('../data/support_materials/cellular_component.txt') as f:
        loc_list = f.read().split()

    ncol = len(loc_list)
    nrow = len(label_list)

    loc_row = []
    loc_col = []
    loc_data = []
    protein_list, loc = zip(*label_list)

    for protein, localizations in tqdm(label_list, desc='construct loc matrix'):
        row = protein_list.index(protein)
        if localizations:
            for localization in localizations:
                col = loc_list.index(localization)
                data = 1
                loc_row.append(row)
                loc_col.append(col)
                loc_data.append(data)

    loc_matrix = coo_matrix((loc_data, (loc_row, loc_col)), shape=(nrow, ncol))

    return loc_matrix


def construct_loc_matrix():
    label_list = extract_localization_data()
    loc_matrix = construct_protein_loc_matrix(label_list)
    label_with_loc = extract_data_with_position(label_list)

    sparse.save_npz('../data/generate_materials/loc_matrix', loc_matrix)
    label_with_loc_path = '../data/generate_materials/label_with_loc_list.json'
    with open(label_with_loc_path, 'w') as f:
        json.dump(label_with_loc, f)
    label_path = '../data/generate_materials/label_list.json'
    with open(label_path, 'w') as f:
        json.dump(label_list, f)

    # vir_label_list = generate_virtual_protein(label_list)
    # vir_loc_matrix = construct_protein_loc_matrix(vir_label_list)
    # sparse.save_npz('../data/generate_materials/vir_loc_matrix', vir_loc_matrix)
    # vir_label_path = '../data/generate_materials/vir_label_list.json'
    # with open(vir_label_path, 'w') as f:
    #     json.dump(vir_label_list, f)


def construct_graph_data(ppi, gcn, label, vir_label):
    interaction_idx = list(zip(ppi.row, ppi.col))
    protein = list(list(zip(*label))[0])
    vir_labal = list(list(zip(*vir_label))[0])
    interaction = set()
    gcn = gcn.toarray()

    for item in tqdm(interaction_idx, desc='add virtual protein'):
        uni_a = protein[item[0]]
        uni_b = protein[item[1]]
        loc_a = label[item[0]][1]
        loc_b = label[item[1]][1]
        expr_row = protein.index(uni_a)
        expr_col = protein.index(uni_b)
        expr_data = gcn[expr_row][expr_col]

        loc_a_flag = True if len(loc_a) > 1 else False
        loc_b_flag = True if len(loc_b) > 1 else False

        if loc_a_flag and loc_b_flag:
            for i in range(len(loc_a)):
                for j in range(len(loc_b)):
                    a = uni_a + '_' + str(i)
                    b = uni_b + '_' + str(j)
                    a_idx = vir_labal.index(a)
                    b_idx = vir_labal.index(b)
                    inter = (a_idx, b_idx, expr_data)
                    interaction.add(inter)

        elif loc_a_flag or loc_b_flag:
            if loc_a_flag:
                for i in range(len(loc_a)):
                    a = uni_a + '_' + str(i)
                    b = uni_b
                    a_idx = vir_labal.index(a)
                    b_idx = vir_labal.index(b)
                    inter = (a_idx, b_idx, expr_data)
                    interaction.add(inter)

            elif loc_b_flag:
                for j in range(len(loc_b)):
                    a = uni_a
                    b = uni_b + '_' + str(j)
                    a_idx = vir_labal.index(a)
                    b_idx = vir_labal.index(b)
                    inter = (a_idx, b_idx, expr_data)
                    interaction.add(inter)

        else:
            a = uni_a
            b = uni_b
            a_idx = vir_labal.index(a)
            b_idx = vir_labal.index(b)
            inter = (a_idx, b_idx, expr_data)
            interaction.add(inter)

    ppi_row = []
    ppi_col = []
    ppi_data = []
    gcn_row = []
    gcn_col = []
    gcn_data = []
    for item in tqdm(interaction, desc='construct ppi matrix and gcn matrix'):
        ppi_row.append(item[0])
        ppi_col.append(item[1])
        ppi_data.append(1)
        gcn_row.append(item[0])
        gcn_col.append(item[1])
        gcn_data.append(item[2])

    ppi_mat = coo_matrix((ppi_data, (ppi_col, ppi_row)), shape=(len(vir_labal), len(vir_labal)))
    gcn_mat = coo_matrix((gcn_data, (gcn_row, gcn_col)), shape=(len(vir_labal), len(vir_labal)))
    ppi_mat.setdiag(0)
    gcn_mat.setdiag(0)
    ppi_mat.eliminate_zeros()
    gcn_mat.eliminate_zeros()

    ecc_mat = edge_clustering_coefficients(ppi_mat)

    return ppi_mat, ecc_mat, gcn_mat


def extract_data_with_position(label_list):
    uni_list, loc_list = zip(*label_list)
    uni_idx = []
    for item in label_list:
        if item[1]:
            uni = item[0]
            idx = uni_list.index(uni)
            uni_idx.append(idx)

    return uni_idx


if __name__ == '__main__':
    data_dict = {
        1: {
            1: '../data/support_materials/GSE31118_family.soft',
            2: {
                'normal': ['GSM770571', 'GSM770572', 'GSM770574'],
                'intervention': ['GSM770586', 'GSM770589', 'GSM770591']
            }
        },
    }
    construct_matrix_of_normal_and_disease_cond(data_dict, 5000, 20000)
    construct_loc_matrix()  # 加入虚拟蛋白需要修改注释部分

    ppi = sparse.load_npz('../data/generate_materials/PPI_normal.npz')
    ecc = sparse.load_npz('../data/generate_materials/ECC_normal.npz')
    gcn = sparse.load_npz('../data/generate_materials/GCN_normal.npz')
    print(ppi.shape, ecc.shape, gcn.shape)



    # with open('../data/generate_materials/label_list.json') as f:
    #     label = json.load(f)
    # extract_data_with_position(label)



    # add virtual protein
    # ppi = sparse.load_npz('../data/generate_materials/PPI_normal.npz')
    # ecc = sparse.load_npz('../data/generate_materials/ECC_normal.npz')
    # gcn = sparse.load_npz('../data/generate_materials/GCN_normal.npz')
    # with open('../data/generate_materials/label_list.json') as f:
    #     label = json.load(f)
    # with open('../data/generate_materials/vir_label_list.json') as f:
    #     vir_label = json.load(f)
    # ppi_mat, ecc_mat, gcn_mat = construct_graph_data(ppi, gcn, label, vir_label)
    #
    # normal_path = '../data/generate_materials/'
    # sparse.save_npz(normal_path + 'vir_PPI_normal', ppi_mat)
    # sparse.save_npz(normal_path + 'vir_ECC_normal', ecc_mat)
    # sparse.save_npz(normal_path + 'vir_GCN_normal', gcn_mat)
