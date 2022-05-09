import dgl
import torch
import argparse
from train import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.000005, metavar='learning rate')
    parser.add_argument('-f', type=int, default=10, metavar='fold num')
    parser.add_argument('-e', type=int, default=150, metavar='epoch num')
    parser.add_argument('-a', nargs='*', default=[0.1, 0.3], metavar='alpha list')
    parser.add_argument('-b', nargs='*', default=[1, 1.2], metavar='beta list')
    parser.add_argument('-d', type=str, default='cuda', metavar='device')
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)

    lr = args.lr
    fold_num = args.f
    epoch_num = args.e
    alpha_list = list(map(float, args.a))
    beta_list = list(map(float, args.b))
    device = args.d

    log_path = '../data/log/normal/'
    os.mkdir(log_path)
    print('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, beta:{}, device:{}'
          .format(lr, fold_num, epoch_num, alpha_list, beta_list, device))
    with open(log_path + 'txt_log.txt', 'w') as f:
        f.write('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, beta:{}, device:{}\n'
          .format(lr, fold_num, epoch_num, alpha_list, beta_list, device))

    ppi = load_npz('../data/generate_materials/GSE30931_data/PPI_inter.npz')
    gcn = np.load('../data/generate_materials/GSE30931_data/ECC_inter_pca.npy')
    ecc = np.load('../data/generate_materials/GSE30931_data/GCN_inter_pca.npy')
    loc = load_npz('../data/generate_materials/loc_matrix.npz')
    with open('../data/generate_materials/protein_ppi.json', 'r') as f:
        uniprot = json.load(f)

    g = create_graph(ppi, ecc, gcn, loc, uniprot)
    g = g.to(device)
    train(g, lr=lr, fold_num=fold_num, epoch_num=epoch_num, alpha_list=alpha_list, beta_list=beta_list, device=device, path=log_path)

