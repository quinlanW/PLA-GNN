from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import torch as th

if __name__ == "__main__":
    # PPI = load_npz('../data/generate_materials/PPI_normal.npz')
    # LOC = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
    # GCN = load_npz('../data/generate_materials/GCN_normal.npz').tocsr().multiply(PPI.tocsr()).tocoo()
    # ECC = load_npz('../data/generate_materials/ECC_normal.npz')
    # plt.hist(x=GCN.data, bins=100)
    # plt.show()
    # print(len(GCN.data))

    # inter = list(zip(PPI.row, PPI.col))
    # interaction = []
    # loc = LOC.sum(1)
    # # print(loc, loc.shape)
    # for item in inter:
    #     if loc[item[0]] and loc[item[1]]:
    #         # print(loc[item[0]], loc[item[1]])
    #         interaction.append(item)
    #
    # print(interaction)
    # print(len(interaction))

    # a = th.Tensor([
    #     [1, 2, 3],
    #     [4, 5, 6]
    # ])
    #
    # b = a.sum(dim=1).reshape(len(a), 1)
    # print(b)
    # a = a / b
    # print(a)

    LOC = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
    loc_sum = LOC.sum(0)
    print(loc_sum)
    print(loc_sum.shape)
   

 # [25 1495   58  350  365  384   36  469   71  274 1509  974]
