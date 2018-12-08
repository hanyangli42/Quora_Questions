import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import random
from matplotlib import pyplot as plt
import pandas as pd

import time
from tqdm import tqdm
import csv


import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import random
from matplotlib import pyplot as plt
import pandas as pd

import time
from tqdm import tqdm
import csv



def pad_data(x, max_len):
    data = x
    data = np.pad(data,(0, max_len - len(data)), mode = 'constant')
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data, 0)
    # data = torch.unsqueeze(data, 0)
    # data = data.view(1, data.shape[0], data.shape[1])
    return data.float()

# create customized datasets
class DQIDataset(Dataset):
    def __init__(self, q1, q2, y, max_len=271):
        self.q1 = q1
        self.q2 = q2
        self.max_len = max_len

        if y is None:
            self.y = None
        else:
            self.y = torch.from_numpy(y)
    
    def __getitem__(self, ind):

        data1 = pad_data(self.q1[ind], self.max_len)
        data2 = pad_data(self.q2[ind], self.max_len)
        
        if self.y is None:
            # label = torch.tensor(0)
            label = torch.tensor([0]) 
        else:
            label = self.y[ind]
        return data1, data2, label

    def __len__(self):
        return self.q1.shape[0]

def custom_collate(seq_list):
    # B x tuple of (q1, q2, label)
    q1_batch, q2_batch, labels = zip(*seq_list)
    return q1_batch, q2_batch, labels
    
    

class Flatten(nn.Module):
    """
    A simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return torch.squeeze(x)



# ResNet
# citation: used ResNet open source code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.elu(out)

        return out

class DQINetwork(nn.Module):
    def __init__(self): 
        super().__init__()        
        self.embedding_network = Sequential(
          nn.Conv1d(1,32,kernel_size = 5,padding = 0,stride = 2,bias = False),
          nn.ELU(inplace=True),
          BasicBlock(32,32), 
          nn.Conv1d(32,64,kernel_size = 5,padding = 0,stride = 2,bias = False),
          nn.ELU(inplace=True),
          BasicBlock(64,64),  
          nn.Conv1d(64,128,kernel_size = 5,padding = 0,stride = 2,bias = False),
          nn.ELU(inplace=True),
          BasicBlock(128,128), 
          nn.Conv1d(128,300,kernel_size = 5,padding = 0,stride = 2,bias = False),
          nn.ELU(inplace=True),
          BasicBlock(300, 300),
          Flatten()
        )
        self.final_layer = Sequential(
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Linear(100, 2)

            # Flatten()
        )


    def get_embedding(self,x):
        alpha=16
        x_avgpool = self.embedding_network(x).mean(2) # average pooling
        x_norm = F.normalize(x_avgpool, p=2, dim=1) # normalize 
        return alpha * x_norm
        
    def forward(self, q1, q2):
        embed1 = self.get_embedding(q1)
        embed2 = self.get_embedding(q2)
        embed_pair = torch.cat((embed1, embed2), dim=1)
        output = self.final_layer(embed_pair)
        return output

# def cos_similarity(embeddings,ind)
#   similarity = []
#   for i1,i2 in ind:
#       similarity.append(cosine_similarity([embeddings[i1,:]],[embeddings[i2,:]])[0][0])
#   return np.array(similarity)
        
        
# def predict(model, test_loader,test_size,emb_size,test_indx,use_cuda):
#   representations = inference(model, test_loader,test_size,emb_size,use_cuda)
#   pred_test = cos_similarity(representations,test_indx)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

# learning rate auto tuner
def tune_lr(optimizer, epoch_num):
    lr = 0.001 * (0.1 ** (epoch_num // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train_val(network, train_loader, dev_loader, criterion, optimizer, epoch_num, device):
    '''
    train function for one epoch
    '''
    start = time.clock()
    network.train()

    # temp
    # tune_lr(optimizer, epoch)
    total_loss = 0.0
    correct_train = 0.0
    total_train = 0.0

    best_val_acc= 0


    for epoch in tqdm(range(epoch_num)):
        start = time.clock()
        network.train()
        # train
        total_train = 0.0
        correct_train = 0.0
        total_loss = 0.0
        for batch_id, (q1,q2, label) in enumerate(train_loader):
            # move data to GPU
            q1, q2, label = q1.to(device), q2.to(device), label.to(device)
            optimizer.zero_grad()
            
            # # temp 
            # if batch_id > 50:
            #     break

            prediction = network(q1, q2)

            loss = criterion(prediction, label)
            loss.backward()

            optimizer.step()
            
            _, label_pred = torch.max(prediction.data, 1)
            
            batch_loss = loss.item() # loss of the current batch
            total_loss += loss.item() # cumulative loss of current epoch
            total_train += label.size(0)
            correct_train += int((label_pred == label).sum())
            if batch_id % 50 == 0:
                print('[%d, %d] epoch_loss: %.03f, batch_loss: %.03f' % (epoch+1, batch_id+1, total_loss, batch_loss))
                print('Training Acc: %d%%' % (100 * correct_train / total_train))
        elapsed = (time.clock() - start)
        print ('Train time: ', elapsed)

        # eval (validation)
        start = time.clock()
        network.eval()
        with torch.no_grad():
            correct_val = 0.0
            total_val = 0.0

            for batch_id, (q1,q2, label) in enumerate(dev_loader):
                q1, q2, label = q1.to(device), q2.to(device), label.to(device)
                prediction = network(q1, q2)
                _, label_pred = torch.max(prediction.data, 1)
                total_val += label.size(0)
                correct_val += int((label_pred == label).sum())

        val_acc = correct_val/total_val

        elapsed = (time.clock() - start)
        print ('[Epoch:%d]: ' % (epoch+1))
        print ('Validation Accuracy: %.03f, Current Best Accuracy: %.03f' % (val_acc, best_val_acc))
        print ('Validation time: ', elapsed)

        if val_acc > best_val_acc:
            torch.save(network.state_dict(), './val_acc_{}.pt'.format(val_acc))
        best_val_acc = val_acc








