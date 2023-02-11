import numpy as np
import scipy.sparse as sp
import random

from torch_sparse import SparseTensor 

# 生成laplace噪声
def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise

# 对每个cluster生成加过噪音之后的邻接矩阵mat、特征信息x，edge_index中的row_index & col_index
def perturb_adj_lap(cluster, epsilon = 5):
    n_nodes = cluster.x.size()[0]
    n_edges = len(cluster.edge_index[1]) // 2 
    N = n_nodes
    E = n_edges
    
    eps_1 = epsilon * 0.01
    eps_2 = epsilon - eps_1
    
    adj = SparseTensor(row = cluster.edge_index[0], 
                       col = cluster.edge_index[1],
                       value = None, sparse_sizes = (N,N))

    A = sp.tril(adj.to_scipy('csr'), k = -1)
    
    noise = get_noise(noise_type='laplace', size=(N, N), 
                      seed=42, eps=eps_2, delta=1e-5, sensitivity=1) * 1000 ####################################
    noise *= np.tri(*noise.shape, k=-1, dtype=bool)
    
    A += noise
    
    noise_edge = int(get_noise(noise_type= 'laplace', size=1, seed=random.randint(0,100),
                    eps=eps_1, delta=1e-5, sensitivity=1)[0])
    n_edges_keep = E - noise_edge
    n_edges_keep = max(n_edges_keep,1) #-------------------at least 2 edges in each cluster
    print(f'edge number from {E} to {n_edges_keep}')
    
    A_extend = A.A.ravel() # 将邻接矩阵展开成向量
    n_splits = 3
    len_sub = len(A_extend) // n_splits

    # find the largest T(= n_edges_keep) cells of A in each part
    ind_list = [] 
    for i in range(n_splits - 1):
        ind = np.argpartition(A_extend[len_sub*i:len_sub*(i+1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_sub * i)
    ind = np.argpartition(A_extend[len_sub*(n_splits-1):], -n_edges_keep)[-n_edges_keep:] 
    ind_list.append(ind + len_sub * (n_splits - 1))
    
    # merge the largest cells of each part together
    # and again find the largest T(= n_edges_keep) cells 
    ind_subset = np.hstack(ind_list) 
    a_subset = A_extend[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]
    
    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert(col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), 
                        shape=(N,N))
    mat = mat + mat.T
    
    row_idx, col_idx = np.array(row_idx + col_idx), np.array(col_idx + row_idx)
    idx_sort = np.lexsort((col_idx,row_idx))

    row_idx = row_idx[idx_sort]
    col_idx = col_idx[idx_sort]
    x = cluster.x
    
    return mat,x,row_idx,col_idx