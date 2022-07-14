import numpy as np
import random
import json
import torch as th
import networkx as nx
from scipy.sparse import load_npz, coo_matrix
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import id_mapping
import urllib.parse
import urllib.request


if __name__ == "__main__":
    # loc mapping
    loc = {
        'GO:0005938': 'Cell cortex',
        'GO:0005829': 'Cytosol',
        'GO:0015629': 'Actin cytoskeleton',
        'GO:0005794': 'Golgi apparatus',
        'GO:0005783': 'Endoplasmic reticulum',
        'GO:0005730': 'Nucleolus',
        'GO:0005777': 'Peroxisome',
        'GO:0005739': 'Mitochondrion',
        'GO:0005764': 'Lysosome',
        'GO:0005813': 'Centrosome',
        'GO:0005634': 'Nucleus',
        'GO:0005886': 'Plasma membrane'
    }


    with open('../data/validation/DRUG: Bortezomib') as f:
        next(f)
        proteins = f.readlines()

    query_list = []
    for line in proteins:
        line = line.strip().split(',')
        this_p = line[1].split()[0]
        query_list.append(this_p)
        interaction_file = ''
        if line[3]:
            interaction_file = '../data/validation/BIOGRID-GENE-' + str(line[3]) + '-4.4.209.tab3.txt'
            print(this_p, line[3], interaction_file)
            with open(interaction_file) as f:
                next(f)
                interaction_p = f.readlines()
            interaction = []
            for l in interaction_p:
                l = l.split('\t')
                # print(line[3], line[4])
                if l[3] != line[3]:
                    interaction.append(l[3])
                if l[4] != line[3]:
                    interaction.append(l[4])
            query_str = ''
            for item in interaction:
                item = item.strip() + ' '
                query_str += item

            url = 'https://www.uniprot.org/uploadlists/'
            params = {
                'from': 'BIOGRID_ID',
                'to': 'ACC',
                'format': 'tab',
                'query': query_str
            }

            data = urllib.parse.urlencode(params)
            data = data.encode('utf-8')
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req) as f:
                response = f.read().decode('utf-8')

            for item in response.strip().split('\n')[1:]:
                targ_id = item.split()[1]
                query_list.append(targ_id)

    with open('../data/generate_materials/loc_change_record(with labels).txt') as f:
        pred = f.readlines()

    for line in pred[0: 101]:
        for uni in query_list:
            if uni in line:
                print(pred.index(line), uni, line)

    print('+'*20)
    for line in pred[101:]:
        for uni in query_list:
            if uni in line:
                print(pred.index(line), uni, line)