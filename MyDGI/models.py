import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

EPS = 1e-8


class MyDGI(torch.nn.Module):
        def __init__(self, hidden_channels, encoder, cor, n_class):
                            # 需要修改的是corruption
            super().__init__()
            self.hidden_channels = hidden_channels
            self.encoder = encoder
            self.cor = cor 
            self.linear = nn.Linear(hidden_channels, n_class)  #####这个encoder变成两层的话，这里改成self.linear = nn.Linear....
            
        def forward(self, x, edge_index):
            pos_z = self.encoder(x, edge_index) # view a: server_received_x, server_received_edge_index
            cor = self.cor if isinstance(self.cor, tuple) else (self.cor, )
            neg_z = self.encoder(*cor) #view b: server_revised_x, server_revised_edge_index
            y_pred = self.linear(pos_z)         # 返回的是各个类别的概率
            return pos_z, neg_z, y_pred
        
        def discriminate(self, z, summary, sigmoid=True):
            summary = summary.t() if summary.dim() > 1 else summary
            value = torch.matmul(z, torch.matmul(self.weight, summary))
            return torch.sigmoid(value) if sigmoid else value
        
        #增添的函数——By Yah
        def embedding_subgraph(self,z,Cln): 
            z_subg = [z[Cln[i]].mean(dim = 0) for i in range(len(Cln))]
            return torch.vstack(z_subg) # z_subg, tensor: size = n_cluster x n_features (4 x 512)
        
        def similarity_node_subgraph_cal(self, z, z_subg): # z:features of nodes: size: n_nodes x n_features (2708 x 512)
                                                      # z_subg: eatures of subgraphs: size: n_subgraphs x n_features (4 x 512)
            sim_nume = torch.matmul(z, z_subg.T)

            z_norm = torch.norm(z, dim=1)
            z_norm = torch.unsqueeze(z_norm,1)
            z_subg_norm = torch.norm(z_subg, dim=1)
            z_subg_norm = torch.unsqueeze(z_subg_norm,1)
            sim_deno = torch.matmul(z_norm, z_subg_norm.T) + EPS ###########
            similarity = sim_nume/sim_deno 
            return similarity  #size: n_nodes x n_clusters (2708 x 4)
            
        
        def loss_cl_cal(self, z, z_subg, mask):   # mask: tensor, size: n_nodes x n_clusters, 对应位置True表明node与cluster是positive的关系
            N = z.size()[0]
            sim_node_subgraph = self.similarity_node_subgraph_cal(z, z_subg)          
            pos_loss_all = torch.log(1 + torch.exp( - sim_node_subgraph) + EPS)
            pos_loss = (pos_loss_all*mask).sum() 
            neg_loss_all = torch.log(1 + torch.exp(sim_node_subgraph) + EPS)
            neg_loss = (neg_loss_all*(mask == False)).sum() 
            return (pos_loss + neg_loss)/N
                
        
        def loss(self, z_a, z_subg_b, z_b, z_subg_a, y_pred, y, mask, rho):
            #contrastive loss
            loss_cl_1 = self.loss_cl_cal(z_a, z_subg_b, mask) #+ self.loss_cl_cal(z_b, z_subg_a, mask)
            loss_cl_2 = self.loss_cl_cal(z_b, z_subg_a, mask)
            loss_cl = loss_cl_1 +  loss_cl_2
                    
            loss_cn = F.cross_entropy(y_pred, y) 
            
            return loss_cl + rho * loss_cn

        def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
                 multi_class='auto', *args, **kwargs):
            r"""Evaluates latent space quality via a logistic regression downstream
            task."""
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                     **kwargs).fit(train_z.detach().cpu().numpy(),
                                                   train_y.detach().cpu().numpy())
            return clf.score(test_z.detach().cpu().numpy(),
                             test_y.detach().cpu().numpy())

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}({self.hidden_channels})'


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
#         self.prelu2 = nn.PReLU(out_channels)
        

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)
        return x
    
class EncoderDGI(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
    
    
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x