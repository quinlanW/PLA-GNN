
import torch as th

if __name__ == "__main__":
    a = th.tensor([
        [0, 1, 1],
        [0, 0, 1]
    ])
    print(a[:, -1:])