from operator import index
from networkx.classes.function import selfloop_edges
import numpy as np
import torch
import time
import networkx as nx


class Fusion_block(torch.nn.Module):
    def __init__(self, fusion_model,batch_size, num_nodes, device):
        super(Fusion_block, self).__init__()
        
        self.fusion_model = fusion_model
        self.device = device
        
        self.e1_L = torch.nn.Parameter( torch.Tensor(batch_size, num_nodes, num_nodes))
        self.e2_L = torch.nn.Parameter( torch.Tensor(batch_size, num_nodes, num_nodes))
        self.e3_L = torch.nn.Parameter( torch.Tensor(batch_size, num_nodes, num_nodes))

    def forward(self, x1, adj1, x3, adj3):

        ###pagerank_fusion###
        if self.fusion_model == 'pagerank':
            e1 = PageRank_Fusion(x1, adj1).to(self.device)
            x1 = torch.matmul(e1, x1)
            adj1 = torch.transpose(torch.matmul(
                e1, torch.transpose(torch.matmul(e1, adj1), 1, 2)), 1, 2)


            e3 = PageRank_Fusion(x3, adj3).to(self.device)
            x3 = torch.matmul(e3, x3)
            adj3 = torch.transpose(torch.matmul(
                e3, torch.transpose(torch.matmul(e3, adj3), 1, 2)), 1, 2)

        return x1, adj1, x3, adj3



def PageRank_Fusion(x, adj):

    adj = adj.cpu()
    x = x.detach().cpu()
    num_dim = x.shape
    x = x.sum(dim=2) 

    batch_size = len(adj)
    num_Nodes = len(adj[0])
    e_all = torch.zeros((batch_size, num_Nodes, num_Nodes))  
    for i in range(batch_size):
        edgs = []
        w = adj[i]
        feature = list(x[i])
        feature_dic = dict(zip(range(len(feature)), feature))
        
        l = (w >= 0.6).nonzero()  
        l = np.array(l)  
        for n in range(len(l)):  

            edg = tuple(
                (l[n][0], l[n][1], {"weight": w[tuple(l[n])].detach().numpy().tolist()}))
            edgs.append(edg) 
        G = nx.Graph()
        G.add_nodes_from([i for i in range(num_Nodes)])
        G.add_edges_from(edgs)
        pr = nx.pagerank(G, alpha=0.85, nstart=feature_dic, tol = 1e-3)  
        list_as = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        s = [list_as[i][0] for i in range(len(list_as))]  
        e_all[i] = torch.eye(num_Nodes)[s]

    return e_all.float()