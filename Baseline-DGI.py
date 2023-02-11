import os.path as osp

import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.loader import ClusterData
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGraphInfomax

from utils import Fun_Server 
from MyDGI import Encoder, EncoderDGI, MyDGI, GCN

EPS = 1e-8

pp = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'Planetoid')
print(pp)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--num_parts', type=int, default=4)
args = parser.parse_args([])

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path = 'D:/Research/Project/GNN/GCN_dragon1860/data' #Path_Yaoh
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset)
    data = dataset[0]
    
    cluster_data = ClusterData(data, args.num_parts, recursive=False) #将数据分成不同cluster
    
    Cln, cluster, data_server_received, data_server_revised = Fun_Server(data, cluster_data)
                    # Cln: size: args.num_parts, 记录每个cluster对应的是原来的哪些节点
                    # data_server：server端包含信息：(X, edge_index)
                    # _received: server端从各个cluster上接收的信息
                    # _revised: server端根据simularity进行调整过后的信息
    
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index    
            
    model = DeepGraphInfomax(
        hidden_channels=512, encoder=EncoderDGI(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    

    def train():
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(*data_server_received) ####
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def test():
        model.eval()
        z, _, _ = model(data.x, data.edge_index)
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask], max_iter=150)
        return acc

    for epoch in range(1, 301):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    acc = test()
    print(f'Accuracy: {acc:.4f}')


if __name__ == "__main__":
    main(args)
