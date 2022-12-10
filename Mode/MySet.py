import torch.utils.data as Data
from torch.utils.data import Dataset, Sampler


class MySet(Dataset):

    def __init__(self, adj, X, a2, a3, a4, a5, enc_inputs, method_inputs, dec_inputs, dec_outputs):
        self.adj = adj
        self.X = X
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.method_inputs = method_inputs


    def __getitem__(self, idx):
        return self.adj[idx], self.X[idx], self.a2[idx], self.a3[idx], self.a4[idx], self.a5[idx], self.enc_inputs[idx], self.method_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


    def __len__(self):
        return len(self.adj)


class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		
        self.indices = range(len(dataset))	 
        self.count = int(len(dataset) / self.batch_size)  

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count
