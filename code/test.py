import numpy as np
import torch as th
import networkx as nx

if __name__ == "__main__":
    a = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 2, 2, 1]
    ])
    print(a[0] + 1e-3)
    # if np.all(a[0] < a[3]):
    #     print(1)
    # G = nx.from_numpy_matrix(a)
    # print(G.nodes)
    # print(G.edges)
    # G.add_edges_from([(3, 4), (1, 4)])
    # print(G.nodes)
    # print(G.edges)
    # G.nodes[0]['label'] = 1
    # G.nodes[1]['label'] = 0
    # G.nodes[2]['label'] = 0
    # G.nodes[3]['label'] = 0
    # G.nodes[4]['label'] = 1
    # dist = nx.shortest_path(G, 0, 0)
    # print(dist)