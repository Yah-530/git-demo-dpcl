import torch

def fun_pos_mask(data, cluster_data, cluster):
    Y_clusters = []
    class_y = torch.unique(data.y)
    num_parts = len(cluster_data)
    N = data.num_nodes
    
    for i in range(num_parts):
        num_class = [len(cluster_data[i].y[cluster_data[i].y == y]) for y in class_y] 
                                          #不同类别分别出现的次数组成的list，比方说对于cluster_0: [23, 15, 10, 532, 84, 4, 2]
        ind = num_class.index(max(num_class)) #出现最多的那个标签的位置
        y_cluster_i = class_y[ind]
        Y_clusters.append(y_cluster_i) #表明每个cluster分别隶属于哪个label 
    print(Y_clusters) 
    
    y_cln = [torch.tensor([idx for idx, e in enumerate(Y_clusters) if e==y]) for y in class_y] 
                          #表明每个label包含哪些cluster
    
    pos_node_cluster_mask = (torch.ones(N,num_parts) < 0)
    for i in range(N):
        pos_node_cluster_mask[i,cluster[i]] = True #结点i自己所在的cluster是positive的
        if torch.numel(y_cln[data.y[i]]):
            pos_node_cluster_mask[i,y_cln[data.y[i]]] = True
    
    return pos_node_cluster_mask
