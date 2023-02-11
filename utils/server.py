import numpy as np
import torch
import torch.utils.data

from torch_geometric.utils import dense_to_sparse  

from .dp import perturb_adj_lap

def Fun_Server(data, cluster_data):
    N = data.num_nodes 
    num_parts = len(cluster_data)
    
    cluster = torch.zeros([N], dtype=torch.long)
    Cln = [] # 用来判断每个cluster中包含原来哪些nodes, size : cluster数目 x 每一个cluster对应的结点数
    for i in range(num_parts):
        start_index = cluster_data.partptr[i]
        end_index = cluster_data.partptr[i+1]
        cln = cluster_data.perm[start_index:end_index]
        Cln.append(cln)  
        cluster[cluster_data.perm[start_index:end_index]] = i # size: 2708 x 1: 每一个node隶属于哪一个cluster对应的cluster label
    
    # 1. Inter-cluster
    edge_index_server_InterCluster = fun_inter_cluster(data, cluster_data, Cln, cluster)
       #edge_index_server_InterCluster: 用来记录cluster间的连接关系
    
    # 2. Inner-cluster
    Adj_perturbed = []
    X_perturbed = []
    idx_row_perturbed = []
    idx_col_perturbed = []
    for i in range(num_parts):
        mat,x,idx_row, idx_col = perturb_adj_lap(cluster_data[i])
        Adj_perturbed.append(mat)
        X_perturbed.append(x)
        idx_row_perturbed.append(idx_row)
        idx_col_perturbed.append(idx_col)   #received

    # 将每个cluster的feature.X信息传到server的对应位置  
    X_server = torch.zeros(*data.x.shape)
    for i in range(num_parts):
        for j in range(len(Cln[i])):
            X_server[Cln[i][j]] = X_perturbed[i][j]
            
    edge_index_perturbed_revised = [] #revised
    for i in range(num_parts):
        x_i = data.x[Cln[i]]
        idx_row_i = torch.LongTensor(idx_row_perturbed[i])   #idx_row_i与idx_col_i是对称的
        idx_col_i = torch.LongTensor(idx_col_perturbed[i])
        edge_index_server_revised_i = Server_clusteri_edge_Revised(x_i, idx_row_i, idx_col_i, 4)
        edge_index_perturbed_revised.append(edge_index_server_revised_i)
    
    # 3. Server端进行一一对应调整
    edge_index_server_received = Server_concate_cal(idx_row_perturbed, idx_col_perturbed, Cln, edge_index_server_InterCluster)
    data_server_received = (X_server, edge_index_server_received)
    
    idx_row_perturbed_revised = [edge_index_perturbed_revised[i][0] for i in range(len(edge_index_perturbed_revised))]
    idx_col_perturbed_revised = [edge_index_perturbed_revised[i][1] for i in range(len(edge_index_perturbed_revised))]
    edge_index_server_revised = Server_concate_cal(idx_row_perturbed_revised, idx_col_perturbed_revised, Cln, edge_index_server_InterCluster)
    data_server_revised = (X_server, edge_index_server_revised)
    return Cln, cluster, data_server_received, data_server_revised
    
    
def similarity_cal(x):
    sim_nume = torch.matmul(x, x.T)
    
    feature_norm = torch.norm(x, dim = 1)
#     feature_norm = x.sum(dim = 1)   #l1 norm
    feature_norm = torch.unsqueeze(feature_norm,1) # supposed to be l2 norm
    sim_deno = torch.matmul(feature_norm, feature_norm.T)
    similarity = sim_nume/sim_deno
    return similarity 

def fun_inter_cluster(data, cluster_data, Cln, cluster):
    row = data.edge_index[0]
    col = data.edge_index[1]
    
    row_cluster = cluster[row]
    col_cluster = cluster[col]
    IsInterCluster = (row_cluster != col_cluster)   #  type : bool, 判断row，col中的结点是否隶属于不同簇
    edge_index_server_InterCluster = data.edge_index[:, IsInterCluster == True]
    return edge_index_server_InterCluster


def Server_clusteri_edge_Revised(x_i, idx_row_i, idx_col_i, k): # i是决定是第几个cluster， # k是决定到底从多少数中进行edge保留
    similarity_i = similarity_cal(x_i)
    Ni = len(x_i)
    Ei = len(idx_row_i)
    Ki = Ei // 2           # cluster中边的个数，为之后对称做准备

    pos_rand = torch.randint(Ni,(2, k*Ki)) #随机生成一些node pair，现在还不对称
    edge_sample = torch.hstack([pos_rand, pos_rand[[-1,0]]]) #对称了
    edge_sample = torch.hstack([edge_sample, torch.vstack([idx_row_i,idx_col_i])]) #把原来隐私保护以后构建的node pair也加进来
    
    idx_sort_sample = np.lexsort((edge_sample[1], edge_sample[0]))
    edge_sample = torch.vstack([edge_sample[0][idx_sort_sample], edge_sample[1][idx_sort_sample]])
    
    edge_mask_i = (torch.ones(Ni,Ni) < 0)
    for edge in range(edge_sample.size()[1]):
        i = edge_sample[0][edge]
        j = edge_sample[1][edge]
        edge_mask_i[i,j] = True
    for i in range(Ni):
        edge_mask_i[i,i] = False
    sim = similarity_i[edge_mask_i].numpy()
    sim.sort()
    ind = np.argpartition(sim, -Ei)[-Ei:] #寻找sim中最大的Ei个元素
    sim_Ki_max = sim[ind]
    sim_Ki_max.sort()
    sim_thre = sim_Ki_max[0] #阈值，用来比较similarity是否大于阈值，可以保留（是在原来sample的k*Ki个边中寻找保留）
    cluster_edge_mask = (similarity_i*edge_mask_i >= sim_thre)
    edge_index_server_revised_i = dense_to_sparse(cluster_edge_mask)[0]
    
    return edge_index_server_revised_i

def Server_concate_cal(idx_row_perturbed, idx_col_perturbed, Cln, edge_index_server_InterCluster):
    num_parts = len(Cln)
    row_index_server_perturbed = []
    col_index_server_perturbed = []

    for i in range(num_parts):
        idx_row_i = idx_row_perturbed[i]
        idx_row_i = Cln[i][idx_row_i]

        idx_col_i = idx_col_perturbed[i]
        idx_col_i = Cln[i][idx_col_i]
        row_index_server_perturbed.append(idx_row_i)
        col_index_server_perturbed.append(idx_col_i)

    # adding inter-cluster edges 最后加簇间连接

    row_index_server_perturbed.append(edge_index_server_InterCluster[0])
    col_index_server_perturbed.append(edge_index_server_InterCluster[1])

    row_index_server = torch.hstack(row_index_server_perturbed)
    col_index_server = torch.hstack(col_index_server_perturbed)

    #排序
    idx_sort_server = np.lexsort((col_index_server,row_index_server))
    row_index_server = row_index_server[idx_sort_server]
    col_index_server = col_index_server[idx_sort_server]
    edge_index_server = torch.vstack([row_index_server, col_index_server])

    return edge_index_server