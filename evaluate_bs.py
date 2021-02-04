import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import MyLoss
import torch
import gc
import numpy as np


def validate_loss(vali_loader, net,graph,  start,batch_size_v, save, writer, loss_f):
    dataiter = iter(vali_loader)
    input, label = dataiter.next()
    prediction, graph = net(graph, input)
    validation_loss = loss_f(prediction, label, batch_size_v)
    if save:
        writer.add_scalar("validation_loss", validation_loss.item())
        writer.flush()
    del input, label, prediction, dataiter
    gc.collect()
    return validation_loss


def resolution(goal, start, graph,  vali_loader, net ,batch_size_v):
    x = []
    y = []
    z = []
    for n, (input, labels) in enumerate( vali_loader):
        if n > 10:
            break
        prediction, _ = net(graph, input)
        n+=1
        differences = torch.abs(torch.sub(labels, prediction))
        for dif in differences:
            if torch.max(dif) <= goal/2:
                n+=1
            x.append(dif[0].item())
            x.append(dif[3].item())
            x.append(dif[6].item())
            y.append(dif[1].item())
            y.append(dif[4].item())
            y.append(dif[7].item())
            z.append(dif[2].item())
            z.append(dif[5].item())
            z.append(dif[8].item())
        del input, labels, prediction, dif
        gc.collect()
    return n/batch_size_v, [np.mean(np.abs(x)), np.std(x), np.mean(np.abs(y)), np.std(y), np.mean(np.abs(z)), np.std(z)]
