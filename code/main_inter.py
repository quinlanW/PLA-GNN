'''
Perturbation train code

Perturbation condition protein localization prediction file generation
'''
import argparse
import random
from train import *
from utils import *
seed = 70
random.seed(seed)
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)
dgl.seed(seed)
np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, metavar='Dataset used for training\n'
                                                   '\t Available options:\n'
                                                   '\t\tGSE30931'
                                                   '\t\tGSE74572'
                                                   '\t\tGSE27182')
    parser.add_argument('-lr', type=float, default=0.00005, metavar='learning rate')
    parser.add_argument('-f', type=int, default=10, metavar='fold num')
    parser.add_argument('-e', type=int, default=200, metavar='epoch num')
    parser.add_argument('-a', nargs='*', default=[0.1], metavar='alpha list')
    parser.add_argument('-d', type=str, default='cuda', metavar='device')
    args = parser.parse_args()


    path_dict = {
        'GSE30931': '../data/generate_materials/GSE30931_data/',
        'GSE74572': '../data/generate_materials/GSE74572_data/',
        'GSE27182': '../data/generate_materials/GSE27182_data/'
    }

    dataflag = args.data
    dataset = path_dict[dataflag]
    lr = args.lr
    fold_num = args.f
    epoch_num = args.e
    alpha_list = list(map(float, args.a))
    device = args.d

    log_path = '../data/log/' + dataflag + '/perturbation/'
    # log_path = '../data/log/inter_b' + str(int(beta * 10)) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, device:{}'
          .format(lr, fold_num, epoch_num, alpha_list, device))
    with open(log_path + 'txt_log.txt', 'w') as f:
        f.write('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, device:{}\n'
                .format(lr, fold_num, epoch_num, alpha_list, device))

    ppi = load_npz(dataset + 'PPI_inter.npz').tocoo()
    ecc = np.load(dataset + 'ECC_inter_pca.npy')
    gcn = np.load(dataset + 'GCN_inter_pca.npy')
    loc = load_npz('../data/generate_materials/loc_matrix.npz')
    expr = np.load(dataset + 'expr_inter.npy')
    with open('../data/generate_materials/protein_ppi.json', 'r') as f:
        uniprot = json.load(f)

    g = create_graph(ppi, ecc, gcn, loc, expr, uniprot)
    g = g.to(device)
    train(g, lr=lr, fold_num=fold_num, epoch_num=epoch_num, alpha_list=alpha_list, device=device, path=log_path)

