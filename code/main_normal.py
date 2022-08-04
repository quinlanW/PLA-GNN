'''
Normal train
'''
import argparse
from train import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.00005, metavar='learning rate')
    parser.add_argument('-f', type=int, default=10, metavar='fold num')
    parser.add_argument('-e', type=int, default=200, metavar='epoch num')
    parser.add_argument('-a', nargs='*', default=[0.1], metavar='alpha list')
    parser.add_argument('-d', type=str, default='cuda', metavar='device')
    args = parser.parse_args()

    seed = 70
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)

    lr = args.lr
    fold_num = args.f
    epoch_num = args.e
    alpha_list = list(map(float, args.a))
    device = args.d

    beta = 1  # The parameter has been deprecated

    log_path = '../data/log/normal_b' + str(int(beta * 10)) + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, beta:{:},device:{}'
          .format(lr, fold_num, epoch_num, alpha_list, beta, device))
    with open(log_path + 'txt_log.txt', 'w') as f:
        f.write('learning rate:{:.8f}, fold num:{:}, epoch num:{:}, alpha list:{}, beta{:},device:{}\n'
                .format(lr, fold_num, epoch_num, alpha_list, beta, device))

    ppi = load_npz('../data/generate_materials/PPI_normal.npz')
    gcn = np.load('../data/generate_materials/ECC_normal_pca.npy')
    ecc = np.load('../data/generate_materials/GCN_normal_pca.npy')
    loc = load_npz('../data/generate_materials/loc_matrix.npz')
    expr = np.load('../data/generate_materials/expr_normal.npy')
    with open('../data/generate_materials/protein_ppi.json', 'r') as f:
        uniprot = json.load(f)

    g = create_graph(ppi, ecc, gcn, loc, expr, uniprot)
    g = g.to(device)
    train(g, lr=lr, fold_num=fold_num, epoch_num=epoch_num, alpha_list=alpha_list, beta=beta, device=device, path=log_path, seed=seed)
