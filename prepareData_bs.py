import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def from_binary(path):
    f = open(path, 'rb')
    objs = []
    while 1:
        try:
            o = pickle.load(f)
            objs.append(o)
        except EOFError:
            break
    return objs

def loadData(pfad_data, pfad_truth,n,start):
    #graph = dgl.transform.knn_graph(start, 8)
    data = from_binary(pfad_data)
    data_graphs = []
    if len(data) == 1:
        data = data[0]
    #for d in data:
    #    graph.ndata['x'] = torch.tensor(d)
    #    data_graphs.append(graph)
    truth = from_binary(pfad_truth)
    truth = np.reshape(truth,(len(data),9))
    if n > len(data):
        n = len(data)
        print("ERROR: The dataset is too short!")
    data = data[:n]
    list = []
    for i in range(len(data)):
        list.append((data[i], truth[i]))
    return list

class MyData(Dataset):
    def __init__(self,pfad_data, pfad_truth,n,start):
        self.dataset = loadData(pfad_data, pfad_truth,n,start)

    def __getitem__(self, item):
        event, label  = self.dataset[item]
        return event, label

    def __len__(self):
        return len(self.dataset)


import dgl
import torch

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

def giveDataloader(p_train_data,p_train_truth,batch_size_t,n,start):
    return DataLoader(MyData(p_train_data,p_train_truth,n,start),batch_size = batch_size_t,  shuffle=True)#,collate_fn=collate)


######
def loadData2(pfad_data, pfad_truth,n,pfad_data2, pfad_truth2,n2):
    data = from_binary(pfad_data)
    if len(data) == 1:
        data = data[0]
    truth = from_binary(pfad_truth)
    data2 = from_binary(pfad_data2)
    if len(data2) == 1:
        data2 = data2[0]
    truth2 = from_binary(pfad_truth2)
    truth = np.reshape(truth,(len(data),9))
    truth2 = np.reshape(truth2,(len(data2),9))
    truth2 = truth2[n2:]
    truth = np.vstack((truth,truth2))
    data2 = data2[n2:]
    data = np.vstack((data,data2))
    if n > len(data):
        n = len(data)
        print("ERROR: The dataset is too short!")
    list = []
    for i in range(len(data)):
        list.append((data[i], truth[i]))
    return list

class MyData2(Dataset):
    def __init__(self,pfad_data, pfad_truth,n,pfad_data2, pfad_truth2,n2) :
        self.dataset = loadData2(pfad_data, pfad_truth,n,pfad_data2, pfad_truth2,n2)

    def __getitem__(self, item):
        event, label  = self.dataset[item]
        return event, label

    def __len__(self):
        return len(self.dataset)

def giveDataloader2(pfad_data, pfad_truth,pfad_data2, pfad_truth2,batch_size_t,n,n2):
    return DataLoader(MyData2(pfad_data, pfad_truth,n,pfad_data2, pfad_truth2,n2),batch_size = batch_size_t,  shuffle=True,collate_fn=collate)
