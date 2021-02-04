from model_bs import Net, my_loss,MyLoss
from evaluate_bs import validate_loss, resolution
from prepareData_bs import giveDataloader, giveDataloader2
from make_log_bs import log, save_model, str2bool

import torch.nn as nn
import torch
import dgl
import numpy as np
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import *
import argparse
import gc
import torch.autograd.gradcheck


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('save', help='Should this run be saved?')
parser.add_argument('comment', help='Commet about the architecture or something.')
args = parser.parse_args()
save = str2bool(args.save)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

p_train_data = "TestData/morePhotons/batch1_data_normSTD.bin"
p_train_truth = "TestData/morePhotons/batch1_simple_truth.bin"
n_train = 2500

p_vali_data = "TestData/morePhotons/batch3_data_normSTD.bin"
p_vali_truth = "TestData/morePhotons/batch3_simple_truth.bin"
n_vali = 250
batch_size_t = 5
batch_size_v = 5
learning_rate = 0.00025
epochs = 20
sigma = 0.01
Dropout = 0.2


positions = np.load("TestData/positions.npy")
start = torch.tensor(positions)
train_loader =   giveDataloader(p_train_data,p_train_truth,batch_size_t, n_train,start)#2(p_train_data,p_train_truth,p_vali_data,p_vali_truth,batch_size_t,n_train,n_vali)#
vali_loader =  giveDataloader(p_vali_data,p_vali_truth,batch_size_v, n_vali,start)
print("%i train events" %n_train)
print("%i validation events" %n_vali)

net = Net(50,10, Dropout)
net = net.float().to(device)
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

if save:
    my_log = log()
    log.writelog_file_start(my_log, args.comment, p_train_data, p_vali_data, batch_size_t, batch_size_v, learning_rate, trainable_params, sigma, Dropout)
    writer = SummaryWriter(my_log.give_dir())
else:
    writer = 0


loss_f = MyLoss(sigma)
graph = dgl.transform.knn_graph(start, 8)
graph  = dgl.batch([ graph for i in range(batch_size_t)])
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
net.train()


dataiter = iter(train_loader)
input, label = dataiter.next()
prediction, _ = net(graph, input)
res = torch.autograd.gradcheck(MyLoss(sigma),(prediction.double(), label.double(), batch_size_t), raise_exception=True)
print("Gradcheck of loss function: " +str(res))
del input, label#, res
gc.collect()


all_loss = []
all_m_loss = []
all_Vloss = []
all_m_Vloss = []
all_diff =[0,0,0,0,0,0]
all_learned = 0
overfit = 0
for epoch in range(epochs):
    print('Epoch %d' %epoch)
    m_loss = []
    m_Vloss = []
    for i, (input, label) in enumerate( tqdm(train_loader)):
        prediction, graph = net(graph, input)
        loss = loss_f(prediction, label, batch_size_t)# my_loss(prediction[0], label, sigma)#
        m_loss.append(loss.detach().numpy())
        if save:
            writer.add_scalar("loss", loss.item())
        vall_loss = validate_loss(vali_loader, net, graph, start, batch_size_v, save, writer, loss_f)
        m_Vloss.append(vall_loss.detach().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del input, label, prediction, i, loss, vall_loss
        gc.collect()
    all_m_loss.append(np.mean(m_loss))
    all_loss.append(m_loss)
    all_m_Vloss.append(np.mean(m_Vloss))
    all_Vloss.append(m_Vloss)
    res, diff = resolution(0.2, start, graph, train_loader, net, batch_size_v)
    print('Loss: %.4f  | Validation Loss: %.4f' % (np.mean(m_loss), np.mean(m_Vloss)))
    print('mean Distances (Pred - Truth) x: %.4f +/- %.4f, y: %.4f +/- %.4f, z: %.4f +/- %.4f'  %(diff[0],diff[1],diff[2],diff[3],diff[4],diff[5]))
    all_diff = np.vstack((all_diff, diff))
    del res, diff
    gc.collect()
    if len(all_m_loss) > 1:
        if all_m_loss[-1] == all_m_loss[-2]:
            all_learned +=1
            if all_learned > 2:
                epochs = epoch +1
                print("Break up training because mean loss does not change, twice.")
                break
        else:
            all_learned = 0
        if all_m_Vloss[-1]*0.9 > all_m_loss[-1]:
            overfit +=1
            if overfit >1:
                epochs = epoch+1
                print("Break up training because of overfitting.")
                break
        else:
            overfit = 0
    save_model(net, my_log.give_dir(), epoch)


all_diff = all_diff[1:]
if save:
    writer.close()
    log.writelog_file_end(my_log, epochs, all_loss, all_Vloss, all_diff)
    save_model(net, my_log.give_dir(), epoch)
    log.save_plots(my_log, epochs, n_train/batch_size_t, np.ravel(all_loss), all_m_loss, np.ravel(all_Vloss), all_m_Vloss, all_diff)
