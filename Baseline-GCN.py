import os.path as osp

import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.loader import ClusterData
from torch_geometric.datasets import Planetoid

from utils import Fun_Server 
from MyDGI import Encoder, MyDGI, GCN

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
                    
    model = GCN(dataset.num_features, 16, num_class).to(device)
    # data_server_received = data_server_received.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr = 0.01)
    

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data_server_received[0], data_server_received[1])
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)
    
    def test():
        model.eval()
        pred = model(data_server_received[0], data_server_received[1]).argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs

    best_val_acc = 0
    for epoch in range(1, 300 + 1):
        loss = train()
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    print(f'Accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main(args)
