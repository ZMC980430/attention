import torch
from torch import nn
import matplotlib.pyplot as plt
import d2l.torch


def heatmap(matrices, xlabel, ylabel, cmap='Reds'):
    rows, cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, squeeze=False)
    for col_axes, col_matrices in zip(axes, matrices):
        for ax, matrix in zip(col_axes, col_matrices):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.colorbar(pcm, ax=axes)


# unfinished
def masked_softmax(matrix, length=0):
    if length == 0:
        return nn.functional.softmax(matrix, -1)
    if isinstance(length, torch.Tensor):
        length = [int(l) for l in length.flatten().numpy()]
    if isinstance(length, int):
        matrix[..., length:] = torch.ones(matrix.shape[-1]-length)*-1e6
    return nn.functional.softmax(matrix, dim=-1)

if __name__ == '__main__':
    a=torch.rand(2,3,4,5)
    b=torch.Tensor([3])
    print(masked_softmax(a, b))
