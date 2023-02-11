import os.path as osp

import argparse
import torch
import torch.utils.data
from torch_geometric.loader import ClusterData
from torch_geometric.datasets import Planetoid

from utils import Fun_Server, fun_pos_mask
from MyDGI import Encoder, MyDGI

EPS = 1e-8

pp = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
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
    num_class = len(torch.unique(data.y))
    
    cluster_data = ClusterData(data, args.num_parts, recursive=False) #将数据分成不同cluster
    
    Cln, cluster, data_server_received, data_server_revised = Fun_Server(data, cluster_data)
                    # Cln: size: args.num_parts, 记录每个cluster对应的是原来的哪些节点
                    # data_server：server端包含信息：(X, edge_index)
                    # _received: server端从各个cluster上接收的信息
                    # _revised: server端根据simularity进行调整过后的信息
                    
    model = MyDGI(hidden_channels=64, encoder=Encoder(dataset.num_features,512,64),
              cor = data_server_revised, n_class = num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    y_true = torch.unsqueeze(data.y,1)
    pos_node_cluster_mask = fun_pos_mask(data, cluster_data, cluster)

    def train():
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, y_pred = model(*data_server_received) 
        pos_z_subg = model.embedding_subgraph(pos_z,Cln)
        neg_z_subg = model.embedding_subgraph(neg_z,Cln)
        loss = model.loss(pos_z, pos_z_subg, neg_z, neg_z_subg, y_pred[data.train_mask], data.y[data.train_mask], pos_node_cluster_mask, 0.7)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def test():
        pos_z, neg_z, y_pred = model(*data_server_received) 
        acc1 = model.test(pos_z[data.train_mask], data.y[data.train_mask], pos_z[data.test_mask], data.y[data.test_mask], max_iter = 150)
        y_pred = torch.unsqueeze(y_pred.argmax(dim=-1),1)
        acc2 = (y_pred[data.test_mask] == y_true[data.test_mask]).sum()/len(data.y[data.test_mask])
        return acc1, acc2

    for epoch in range(1,400):
        loss = train()
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 20 == 0:
            acc1,acc2 = test()
            print('------------------Accuracy---------------------')
            print(f'Accuracy-1: {acc1:.4f}, Accuracy-2: {acc2:.4f}')
            print('-----------------------------------------------')
            
    acc1,acc2 = test()
    print(f'Accuracy-1: {acc1:.4f}, Accuracy-2: {acc2:.4f}')


if __name__ == "__main__":
    main(args)
