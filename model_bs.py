import dgl
from dgl import backend as B
from scipy import sparse
#from dgl.transform import knn_graph
import torch.nn as nn
import torch
from dgl.nn.pytorch import EdgeConv, GraphConv, GINConv
from dgl.nn.pytorch.glob import MaxPooling
import torch.nn.functional as F
import numpy as np
import gc
from torch.nn.modules.module import Module

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    print("I flatten")
    return t

def prepareInput(input):
    sum = torch.sum(input, axis = -1)
    max = torch.max(input, axis = -1)[0]
    mean = torch.mean(input, axis = -1)
    std = torch.std(input, axis = -1)
    feat = torch.stack((torch.tensor(input[:,0]).clone().detach(),torch.tensor(input[:,-1]).clone().detach(), sum.detach(), max.detach(),mean.detach(),std.detach()), dim = -1)#)
    return feat

def simpleStart(input,start):
        dat = np.array(input)
        data_firsthit = dat[:,0]
        wo_f = np.where(data_firsthit==np.max(data_firsthit))[0][0]
        wo = start[wo_f]
        return wo

class Net(nn.Module):
    def __init__(self, n_feats_fc, in_feats_g, Dropout):
        super(Net, self).__init__()
        self.knn = dgl.nn.pytorch.factory.KNNGraph(8)
        self.edge1 =  EdgeConv(50, 100)
        self.edge2 =  EdgeConv(100, 200)
        self.edge3 = EdgeConv(200, 600)
        self.Dropout = nn.Dropout(Dropout)
        self.pooling = MaxPooling()
        self.fc1 = nn.Linear(600,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 =  nn.Linear(300,100)
        self.fc4 = nn.Linear(100,50)
        self.fc_out = nn.Linear(50, 9)

    def forward(self, graph, inputs):
        inputs = torch.reshape(inputs, (inputs.shape[0]*inputs.shape[1],50))
        feat = torch.tanh(self.edge1(graph, inputs.float()))
        feat = self.Dropout(feat)
        feat = torch.tanh(self.edge2(graph, feat))
        feat = self.edge3(graph, feat)
        feat = torch.max(feat, dim = 1)[0]
        #feat = self.pooling(graph, feat)
        feat = torch.reshape(feat, (int(feat.shape[0]/600), 600))
        feat = torch.tanh(self.fc1(feat))
        feat = self.Dropout(feat)
        feat = torch.tanh(self.fc2(feat))
        feat = torch.tanh(self.fc3(feat))
        feat = torch.tanh(self.fc4(feat))
        out = torch.clamp(self.fc_out(feat), min=-2, max=2)
        return out, graph

def abstand(p, t):
    a = torch.sqrt(torch.sum((p-t)**2))
    return a

def abstand_a(p, t):
    a = torch.sqrt(torch.sum((p-t)**2))+ 10*(p[2]-2)
    return a

def v_multi(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

def v_betrag(v1):
    return np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)

def collinear(p1,p2,p3):
    vektor1 = p2-p1
    vektor2 = p3-p2
    z = (torch.sqrt(torch.sum((vektor1)**2))*torch.sqrt(torch.sum((vektor2)**2)))
    if z == 0:
         z = 0.0001
    corr = (torch.abs(torch.sum((vektor1*vektor2)**2)/z))
    return  corr

def my_loss(pred, tru, sigma):
    loss = torch.tensor([])
    for p,t in list(zip(pred, tru)):
        l = (abstand(p[:3],t[:3])/sigma)**2 + (abstand(p[3:6],t[3:6])/sigma)**2+ (abstand(p[6:],t[6:])/sigma)**2 + (torch.abs(1-collinear(p[:3],p[3:6],p[6:])))**2
    return l


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class MyLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self,sigma):
        super(MyLoss, self).__init__()
        self.sigma = sigma

    def forward(self, p, t, batchsize):
        print(p)
        #loss = torch.tensor([0])
        sigma = self.sigma
        #dif = torch.sum(torch.abs(p[0]-p[1]))
        l = (abstand(p[:,:3],t[:,:3])/sigma)**2 + (abstand(p[:,3:6],t[:,3:6])/sigma)**2+ (abstand(p[:,6:],t[:,6:])/sigma)**2 + (torch.abs(1-collinear(p[:,:3],p[:,3:6],p[:,6:])))**2 #+torch.div(1,dif)
        #loss = torch.add(loss,l)
        #print( torch.div(loss,batchsize),torch.div(1,dif))
        out = torch.div(l,batchsize)
        return out
