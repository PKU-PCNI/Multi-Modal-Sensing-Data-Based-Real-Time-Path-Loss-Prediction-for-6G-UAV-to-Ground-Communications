import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.io as sciio
from torchsummary import summary
import time
import numpy as np
from torch.utils.data import DataLoader
from data_feed_Sequence_new2 import DataFeed

dropout1 = 0
dropout2 = 0
class Linear_net(nn.Module):
    def __init__(self):
        super(Linear_net, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(42, 256),   #输入是3*14, yiceng dropout   # 加d、改划分、Adam #加decay、加层数
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(dropout2)
        )
        self.output_layer = nn.Linear(128, 52)
    def forward(self, x):
        x = self.layer(x)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x)
        return output

test_dir = 'E:\\Desktop\\WCL_code_git\\test'
BATCH_SIZE = 1
Test_loader = DataLoader(DataFeed(test_dir,nat_sort=True),batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
clone = Linear_net()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
loss_func = nn.MSELoss()
#验证开始
print('start test')
total_loss = 0
n = 0
with torch.no_grad():
    clone.eval()
for step, (building, label) in enumerate(Test_loader):
    # 将训练数据移到GPU上
    temp = step + 1
    n = n+1
    building = torch.reshape(building, (BATCH_SIZE, 42))
    building = building.to(torch.float32)
    label = torch.squeeze(label, 1)
    #label = label.cuda()
    output = clone(building)
    '''inputpath = 'E:\Desktop\Image_pl_Scenario_WI\input_test\input_test' + str(temp) + '.mat'
    inputmat = building.tolist()
    sciio.savemat(inputpath, {'input_test' + str(temp): inputmat})'''
    labelpath = 'E:\\Desktop\\WCL_code_git\\label_test\\label_test'+ str(temp) +'.mat'
    outputpath = 'E:\\Desktop\\WCL_code_git\\output_test\\output_test'+ str(temp) +'.mat'
    labelmat = label.tolist()
    outputmat = output.tolist()
    sciio.savemat(labelpath, {'label_test':labelmat})
    sciio.savemat(outputpath,{'output_test':outputmat})
    print(output)
    #print(label)
    #loss = loss_func(output, label)/torch.norm(label,p=2)**2
    loss = loss_func(output, label)
    total_loss = total_loss + loss
    #print(total_loss)
    print(loss)
print("average MSE loss = %.4e" %(total_loss/418))

print(n)