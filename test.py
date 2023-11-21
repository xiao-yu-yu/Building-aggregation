import  torch
label=torch.tensor([0,0,1,1,0,0,1,0])
V=label.size(0)
label_count = torch.bincount(label)
label_count = label_count[label_count.nonzero()].squeeze()
cluster_sizes = torch.zeros(2).long()
cluster_sizes[torch.unique(label)] = label_count
weight = [(V - cluster_sizes).float()]*[(V - cluster_sizes).float()] / V
weight *= (cluster_sizes > 0).float()
print(weight)