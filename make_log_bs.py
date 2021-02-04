import os
import datetime
from shutil import copyfile
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class log:
    def __init__(self):
        self.path = os.getcwd()
        i = 0
        while True:
            name = "improve_Architecture/Train_26/in_Batch/Train_"+str(i)
            try:
                os.mkdir( self.path +"/" + name)
            except OSError:
                pass
            else:
                print ("Successfully created the directory %s%s " % (self.path,name))
                break
            i+=1
        self.dir = self.path +"/"+ str(name)

    def give_dir(self):
        return self.dir

    def writelog_file_start(self, comment, path_dataset, path_validataset ,batch_size_t, batch_size_v,trainable_params, learning_rate, sigma, Dropout):
        copyfile(self.path + "/model_bs.py",self.dir + "/model_bs.py")
        file = open(self.dir + "/log.txt", "a")
        file.write("Training neuronal network" +"\n")
        file.write("For network architecture look at model.py" +"\n")
        file.write(comment +"\n")
        file.write("start time " + str(datetime.datetime.now()) +"\n")
        file.write("Train Data " + path_dataset +"\n")
        file.write("Train Data " + path_validataset +"\n")
        file.write("Batchsize: training: " + str(batch_size_t) + " Validation: " + str(batch_size_v)  +"\n" )
        file.write("Trainable Parameters : " + str(trainable_params) +"\n")
        file.write("Learning rate " + str(learning_rate) +"\n")
        file.write("Loss Function: My Loss Ohne Collinear"+ "sigma= "+ str(sigma)+"\n")
        file.write("Dropout: "+  str(Dropout)+"\n")

    def writelog_file_end(self, epochs, Loss, vali_Loss, all_diff):
        file = open(self.dir + "/log.txt", "a")
        file.write("\n")
        file.write("Epochs " + str(epochs) +"\n")
        file.write("Loss " + str(Loss) +"\n")
        file.write("Validation Loss " + str(vali_Loss) +"\n")
        file.write("Differences Prediction - Truth " + str(all_diff)  +"\n")
        file.write("end time " + str(datetime.datetime.now()) +"\n")
        file.close()

    def save_plots(self, epochs,n, all_loss, all_m_loss, all_Vloss, all_m_Vloss, all_diff):
        #plt.rcParams.update({'font.size': 20})
        x = np.arange(len(all_loss)/epochs/2,len(all_loss)+1, len(all_loss)/epochs)
        x_long = np.arange(0,len(all_loss),1)
        plt.title("Loss vs Validation Loss")
        plt.plot(x_long, all_Vloss, alpha = 0.5, color = "blue", label = "Validation loss")
        plt.plot(x_long, all_loss, alpha = 0.5, color = "orange", label = "Trainings loss")
        plt.plot(x, all_m_Vloss, '--', color = "blue",  label = "mean Validation loss")
        plt.plot(x, all_m_loss,'--', color = "orange", label = "mean Trainings loss")
        plt.legend()
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.grid(True)
        plt.savefig(self.dir + "/losses", format = "pdf")
        fig, axs = plt.subplots(3)
        fig.suptitle('Absolut difference Prediction vs Truth')
        x = np.arange(0,epochs,1)
        axs[0].errorbar(x, all_diff[:,0], yerr = all_diff[:,1], fmt = ".", label = "X")
        axs[1].errorbar(x, all_diff[:,2], yerr = all_diff[:,3], fmt = ".", label = "Y")
        axs[2].errorbar(x, all_diff[:,4], yerr = all_diff[:,5], fmt = ".",label = "Z")
        axs[0].set_ylim(0,2)
        axs[1].set_ylim(0,2)
        axs[2].set_ylim(0,2)
        axs[2].set_xlabel("epochs")
        axs[0].set_ylabel("X")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[1].set_ylabel("Y")
        axs[2].set_ylabel("Z")
        fig.savefig(self.dir + "/guete", format = "pdf")

def save_model(state, path, epoch):
    torch.save(state, path + "/model_"+ str(epoch) +".pth.tar")
