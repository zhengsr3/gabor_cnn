from __future__ import print_function
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from scipy import ndimage as ndi
import random as rd

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import time as tm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
ftrain=open('train.csv','r')
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
class CNNGAB(nn.Module):
    def __init__(self):
        super(CNNGAB, self).__init__()
        self.kernel_para=nn.Parameter(torch.randn(16,2))
        self.dr1=nn.Dropout(0.5)
        self.jh1=nn.ReLU()
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.Dropout(0.5),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        ts=tm.time()
        #print(1)
        row_arr=[[-1,-1,-1],[0,0,0],[1,1,1]]
        col_arr=[[-1,0,1],[-1,0,1],[-1,0,1]]
        row_mat=Variable(torch.FloatTensor(row_arr))
        col_mat=Variable(torch.FloatTensor(col_arr))
        #print(tm.time()-ts)
        para0=self.kernel_para[:,0].repeat(3,3,1).transpose(0,2)
        para1=self.kernel_para[:,1].repeat(3,3,1,1).transpose(0,3).transpose(1,2)
        #print(para1)
        rotrow=row_mat*torch.cos(para0)+col_mat*torch.sin(para0)
        #print('a')
        #print(para0)
        #print('b')
        #print(rotrow)
        #print(rotrow[0,0,0])
        #rotrow[0,0,0].backward(retain_graph=True)
        #print(self.kernel_para.grad.data)
        rotcol=-row_mat*torch.sin(para0)+col_mat*torch.cos(para0)
        #print(rotrow)
        rotrow_new=rotrow.repeat(1,1,1,1).transpose(0,1)
        rotcol_new=rotcol.repeat(1,1,1,1).transpose(0,1)
        #print(rotrow_new)
        kernels_re_old=torch.exp(-0.5 * (rotrow_new ** 2/1600  + rotcol_new ** 2 /2500))/(8*3.1415926)
        kernels_im=kernels_re_old*torch.sin((2 * 3.1415926 * para1 * rotrow_new ))
        '''print('a')
        print(kernels_im[1,0,0,0])
        kernels_im[1,0,0,0].backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)'''
        kernels_re=kernels_re_old*torch.cos((2 * 3.1415926 * para1 * rotrow_new ))
        #print(kernels_re_old)
        #print(tm.time()-ts)
        
        y_re=F.conv2d(x,kernels_re,padding=1)
        '''print('a')
        print(y_re[0,0])
        print(y_re[0,0].sum())
        middle=y_re[0,0].sum()
        middle.backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)'''
        #print(y_re[0][0])
        #print(1)
        #print(tm.time()-ts)
        y_im=F.conv2d(x,kernels_im,padding=1)
        y=(y_im**2+y_re**2+0.0001)**0.5-0.01
        y1=self.dr1(y)
        y2=self.jh1(y1)
        y3=F.max_pool2d(y2,2)
                    
            
        x1 = self.conv2(y3)
        #print(x[0][0])
        x2 = x1.view(x1.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x2)
        #print(output)
        return output
    def test(self,x,sam_y):
        ts=tm.time()
        #print(1)
        row_arr=[[-2,-2,-2,-2,-2],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2]]
        col_arr=[[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2]]
        row_mat=Variable(torch.FloatTensor(row_arr))
        col_mat=Variable(torch.FloatTensor(col_arr))
        #print(tm.time()-ts)
        para0=self.kernel_para[:,0].repeat(5,5,1).transpose(0,2)
        para1=self.kernel_para[:,1].repeat(5,5,1,1).transpose(0,3).transpose(1,2)
        #print(para1)
        rotrow=row_mat*torch.cos(para0)+col_mat*torch.sin(para0)
        #print('a')
        #print(para0)
        #print('b')
        #print(rotrow)
        #print(rotrow[0,0,0])
        #rotrow[0,0,0].backward(retain_graph=True)
        #print(self.kernel_para.grad.data)
        rotcol=-row_mat*torch.sin(para0)+col_mat*torch.cos(para0)
        #print(rotrow)
        rotrow_new=rotrow.repeat(1,1,1,1).transpose(0,1)
        rotcol_new=rotcol.repeat(1,1,1,1).transpose(0,1)
        #print(rotrow_new)
        kernels_re_old=torch.exp(-0.5 * (rotrow_new ** 2/1600  + rotcol_new ** 2 /2500))/(8*3.1415926)
        kernels_im=kernels_re_old*torch.sin((2 * 3.1415926 * para1 * rotrow_new ))
        middle=kernels_im.sum()
        '''print('a')
        print(kernels_im[1,0,0,0])
        kernels_im[1,0,0,0].backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)'''
        kernels_re=kernels_re_old*torch.cos((2 * 3.1415926 * para1 * rotrow_new ))
        #print(kernels_re_old)
        #print(tm.time()-ts)
        
        y_re=F.conv2d(x,kernels_re,padding=2)
        '''print('a')
        print(y_re[0,0])
        print(y_re[0,0].sum())
        middle=y_re.sum()/10000
        middle.backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)'''
        #print(y_re[0][0])
        #print(1)
        #print(tm.time()-ts)
        y_im=F.conv2d(x,kernels_im,padding=2)
        '''print('a')
        middle=y_im.sum()/10000
        middle.backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)'''
        y=(y_im**2+y_re**2+0.0001)**0.5-0.01
        print('a')
        print(y[0,0])
        print(y[0,0].sum())
        middle=y.sum()
        middle.backward(retain_graph=True)
        print('b')
        print(self.kernel_para.grad.data)
        y1=self.dr1(y)
        y2=self.jh1(y1)
        '''print(y2.requires_grad)
        print('a')
        middle=y2.sum()
        print(middle)
        middle.backward(retain_graph=True)
        print(y2.grad)'''
                    
            
        x1= self.conv2(y)
        #print(x)
        #print(x[0][0])
        x2 = x1.view(x1.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        #print(x)
        '''output = self.out(x)
        summ=output.sum()/300
        summ.backward(retain_graph=True)
        print(summ)
        print(y_im.grad)'''
        #print(output)
        #print(sam_y)
        '''loss2=F.cross_entropy(output,sam_y)
        #print(loss2)
        loss2.backward(retain_graph=True)
        print(output.grad)'''
        #print(output)
        return output
def ran_choose(x,y,num):
    choosen=[]
    sam_x=Variable(torch.FloatTensor(num,1,28,28))
    sam_y=Variable(torch.LongTensor(num))
    j=0
    while j<num:
        #print(j)
        ran_num=rd.randint(0,9999)
        if ran_num not in choosen:
            sam_x[j]=x[ran_num]
            sam_y[j]=y[ran_num]
            choosen.append(ran_num)
            j+=1
    #print(sam_x)
    return([sam_x,sam_y])
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,download=True)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)

# 为了节约时间, 我们测试时只测试前2000个
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:28000]/256
#print(test_x)# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:28000]

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("test.csv").values
target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)/256
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
#print(train.shape)
#print(test.shape)
train_v=Variable(torch.from_numpy(train[:42000]).float())
target_v=Variable(torch.from_numpy(target[:28000]).long())
#print(target_v)
cnn2 = CNNGAB()
optimizer3=op.Adam(cnn2.parameters(),lr=0.01)
loss_func3=nn.CrossEntropyLoss()
st=tm.time()
for i in range(1001):
    [sam_x,sam_y]=ran_choose(train_v,target_v,500)
    output=cnn2(sam_x)
    #cnn2.test(sam_x,sam_y)
    #print(output)
    loss2=loss_func3(output,sam_y)
    print(i)
    print(tm.time()-st)
    print(loss2)
    optimizer3.zero_grad()
    loss2.backward(retain_graph=True)
    #print(cnn2.kernel_para)
    #print(cnn2.kernel_para.grad)
    optimizer3.step()
    if i%100==0:
        test_output = cnn2(test_x[:1000])
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print(pred_y[:10], 'prediction number')
        print(test_y[:10].numpy(), 'real number')
        accracy=1000
        for i in range(1000):
            if pred_y[i]!=test_y[i]:
                accracy-=1
        print(accracy/1000)
test_output = cnn2(test_x[:1000])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y[:10], 'prediction number')
print(test_y[:10].numpy(), 'real number')
accracy=1000
for i in range(1000):
    if pred_y[i]!=test_y[i]:
        accracy-=1
print(accracy/1000)
