from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import Data

from torch_geometric.data import DenseDataLoader, batch
from torch_geometric.nn import DenseGCNConv, dense_diff_pool




class Cons_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, improved=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.alpha = 0.5
        self.improved = improved
        torch.nn.init.xavier_normal_(self.weight)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        I = torch.eye(adj.shape[1], device=adj.device).repeat(len(adj),1 , 1)
        c_adj = self.alpha * ((self.alpha + 1) * I + adj)
        out = torch.matmul(c_adj, out)

        if self.bias is not None:
            out = out + self.bias

        return out



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True, CONV = 'CGCN'):
        super(GNN, self).__init__()

        self.add_loop = add_loop
        self.Conv_dict = {'CGCN':[Cons_GraphConv(in_channels, hidden_channels, normalize=normalize),
                                  Cons_GraphConv(hidden_channels, hidden_channels, normalize=normalize),
                                  Cons_GraphConv(hidden_channels, out_channels, normalize=normalize)],
                          'GCN' :[DenseGCNConv(in_channels, hidden_channels, normalize), 
                                  DenseGCNConv(hidden_channels, hidden_channels, normalize),
                                  DenseGCNConv(hidden_channels, out_channels, normalize)]    
                                }

        self.conv1 = self.Conv_dict[CONV][0]
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = self.Conv_dict[CONV][1]
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = self.Conv_dict[CONV][2]
        self.bn3 = torch.nn.BatchNorm1d(out_channels)


        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x



class Net(torch.nn.Module):
    """branch of feature level fusion_model"""
    def __init__(self, CONV):
        super(Net, self).__init__()

        num_nodes = 50
        self.gnn1_pool = GNN(256, 100, num_nodes, add_loop=True, CONV=CONV)
        self.gnn1_embed = GNN(256, 100, 100, add_loop=True, lin=False, CONV=CONV)

        num_nodes = ceil(0.55 * num_nodes)

        self.gnn2_pool = GNN(3 * 100, 200, num_nodes, CONV=CONV)
        self.gnn2_embed = GNN(3 * 100, 100, 100, lin=False, CONV=CONV)


    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        return x, adj, l1 + l2, e1 + e2