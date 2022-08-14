#!/usr/bin/env python
# coding: utf-8

# # 資料讀取與前處理

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
Path = './anime_faces/'
FILES = os.listdir(Path)
files = list()
labels = list()
w = 10000
l = 10000
for jdx in range(len(FILES)):
    file = torch.from_numpy(np.array((Image.open(Path + FILES[jdx])).resize((64, 64))))
    file = file.permute(2, 0, 1)
    w = file.shape[1] if file.shape[1] < w else w
    l = file.shape[2] if file.shape[2] < l else l
    files.append(file)
    print('\r{2}/{3}'.format(0, 0,jdx + 1, len(FILES)), end = '')


# In[2]:


image = files[0]
N1 = image.shape[0]; N2 = image.shape[1]; N3 = image.shape[2]
images = image.reshape(1, N1, N2, N3)
for i in range(1, len(files)): #len(files)
    images = torch.cat((images, files[i].reshape(1, 3, 64, 64)), 0)
    print('\r{2}/{3}'.format(0, 0, i, len(files)), end = '')


# In[3]:


import matplotlib.pyplot as plt
images = images/255
images = (images - 0.5)/0.5


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[5]:


images = images.to(device = 'cuda')


# # DCGAN模型架構

# # ![image.png](attachment:image.png)

# 輸入值的大小為(nz, ngf, nc)，其中nz為輸入向量的長度，ngf為在generator中傳遞feature maps的大小，nc為顏色通道(RGB)。

# In[6]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[7]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1 = nn.Conv2d(in_channels = 3,
                             out_channels = 64, # 64
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False,
                             )
        self.L2 = nn.Conv2d(in_channels = 64,
                             out_channels = 128, # 128
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False,
                             )
        self.l2 = nn.BatchNorm2d(128, device = 'cuda')
        self.L3 = nn.Conv2d(in_channels = 64*2,
                             out_channels = 256, # 256
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False,
                             )
        self.l3 = nn.BatchNorm2d(256, device = 'cuda')
        self.L4 = nn.Conv2d(in_channels = 64*4,
                             out_channels = 512, # 512
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False,
                             )
        self.l4 = nn.BatchNorm2d(512, device = 'cuda')
        self.L5 = nn.Conv2d(in_channels = 64*8,
                             out_channels = 1, # 1
                             kernel_size = 4,
                             stride = 1,
                             padding = 0,
                             bias = False,
                             )
    def forward(self, x):
        X = self.L1(x)
        X = nn.LeakyReLU(negative_slope = 0.2, inplace = True)(X)
        ##
        X = self.L2(X)
        X = self.l2(X)
        X = nn.LeakyReLU(negative_slope = 0.2, inplace = True)(X)
        ##
        X = self.L3(X)
        X = self.l3(X)
        X = nn.LeakyReLU(negative_slope = 0.2, inplace = True)(X)
        ##
        X = self.L4(X)
        X = self.l4(X)
        X = nn.LeakyReLU(negative_slope = 0.2, inplace = True)(X)
        ##
        X = self.L5(X)
        X = nn.Sigmoid()(X)
        return X


# In[8]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.L1 = nn.ConvTranspose2d(in_channels = 100,
                                     out_channels = 64 * 8,
                                     kernel_size = 4,
                                     stride = 1,
                                     padding = 0,
                                     bias = False,
                                     )
        self.l1 = nn.BatchNorm2d(64*8, device = 'cuda')
        self.L2 = nn.ConvTranspose2d(in_channels = 64 * 8,
                                     out_channels = 64 * 4,
                                     kernel_size = 4,
                                     stride = 2,
                                     padding = 1,
                                     bias = False,
                                     )
        self.l2 = nn.BatchNorm2d(64*4, device = 'cuda')
        self.L3 = nn.ConvTranspose2d(in_channels = 64 * 4,
                                     out_channels = 64 * 2,
                                     kernel_size = 4,
                                     stride = 2,
                                     padding = 1,
                                     bias = False,
                                     )
        self.l3 = nn.BatchNorm2d(64*2, device = 'cuda')
        self.L4 = nn.ConvTranspose2d(in_channels = 64 * 2,
                                     out_channels = 64,
                                     kernel_size = 4,
                                     stride = 2,
                                     padding = 1,
                                     bias = False,
                                    )
        self.l4 = nn.BatchNorm2d(64, device = 'cuda')
        self.L5 = nn.ConvTranspose2d(in_channels = 64,
                                     out_channels = 3,
                                     kernel_size = 4,
                                     stride = 2,
                                     padding = 1,
                                     bias = False,
                                    )
    def forward(self ,x):
        X = self.L1(x)
        X = self.l1(X)
        X = nn.ReLU()(X)
        ##
        X = self.L2(X)
        X = self.l2(X)
        X = nn.ReLU()(X)
        ##
        X = self.L3(X)
        X = self.l3(X)
        X = nn.ReLU()(X)
        ##
        X = self.L4(X)
        X = self.l4(X)
        X = nn.ReLU()(X)
        ##
        X = self.L5(X)
        X = nn.Tanh()(X)
        return X


# # 模型訓練

# In[9]:


def Draw(array):
    plt.imshow((array*0.5 + 0.5).permute(1,2,0).detach().to(device = 'cpu'))
    plt.axis('off')


# In[10]:


GEN = Generator().cuda(0)
GEN.apply(weights_init)
DIS = Discriminator().cuda(0)
DIS.apply(weights_init)
LOSS = nn.BCELoss()
GOPT = torch.optim.Adam(GEN.parameters(), lr = 0.0001, betas = (0.5, 0.999))
DOPT = torch.optim.Adam(DIS.parameters(), lr = 0.0001, betas = (0.5, 0.999))
losses = list()


# In[11]:


batches = 32
photo_show = torch.randn((5, 100, 1, 1)).detach().to(device = 'cuda')
k = 0
for epoch in range(100):
    D_total = 0
    G_total = 0
    for batch in range(0, images.shape[0], batches):
        if batch + batches > images.shape[0]:
            train = images[batch:]
            BATCH = images.shape[0] - batch
        else:
            train = images[batch:batch + batches]
            BATCH = batches
    #I = np.random.choice([i for i in range(images.shape[0])], batches, replace = False)
    #BATCH = batches
    #train = images[I]
    ## Discriminator的更新
        DOPT.zero_grad()
        TRUE = torch.tensor(np.ones(shape = (BATCH, ))).to(torch.float32).to(device = 'cuda')
        D_loss1 = LOSS(DIS(train).view(-1), TRUE)
        D_loss1.backward()
        noise = torch.randn(BATCH, 100, 1, 1, device = 'cuda')
        fake = GEN(noise)
        FAKE = torch.tensor(np.zeros(shape = (BATCH, ))).to(torch.float32).to(device = 'cuda')
        D_loss2 = LOSS(DIS(fake.detach()).view(-1), FAKE)
        D_loss2.backward()
        D_loss = D_loss1 + D_loss2
        DOPT.step()
        ## Generator的更新
        GOPT.zero_grad()
        noise.data.copy_(torch.randn(BATCH, 100, 1, 1, device = 'cuda'))
        fake = GEN(noise)
        G_loss = LOSS(DIS(fake).view(-1), 1 - FAKE)
        G_loss.backward()
        GOPT.step()
        B = batch + BATCH
        #print("Epoch: {0} Batch:{1} --Generator Loss: {2}; Discriminator Loss: {3}".format(len(losses) + 1,
        #                                                                                   str(B) + '/' + str(images.shape[0]),
        #                                                                                   float(G_loss),
        #                                                                                   float(D_loss)))
        D_total += float(D_loss)*BATCH/images.shape[0]
        G_total += float(G_loss)*BATCH/images.shape[0]
        if k == 50:
            flag = 1
            print('Epoch:', len(losses) + 1)
            for obj in GEN(photo_show):
                plt.subplot(1, 5, flag)
                Draw(obj.to(device = 'cpu'))
                flag += 1
            plt.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            k = 0    
        k += 1
    losses.append([G_total, D_total])
    if G_loss < 100:
        torch.save({'model1':GEN.state_dict(), 
                    'model2':DIS.state_dict(),
                    'optimizer1':GOPT.state_dict(), 
                    'optimizer2':DOPT.state_dict(),
                    'LOSS':losses}, 
                    'DCGAN.pt')


# In[26]:


OBJ = GEN(torch.randn((25, 100, 1, 1)).detach().to(device = 'cuda'))
for i in range(0, 5):
    for j in range(1, 6):
        plt.subplot(5, 5, i*5 + j)
        Draw(OBJ[i*5 + j - 1].to(device = 'cpu'))


# In[25]:


plt.plot(np.array(losses)[:,0], label = 'G')
plt.plot(np.array(losses)[:,1], label = 'D')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


FILE = torch.load('Generative.pt')
GEN.load_state_dict(FILE['model1'])
DIS.load_state_dict(FILE['model2'])
GOPT.load_state_dict(FILE['optimizer1'])
DOPT.load_state_dict(FILE['optimizer2'])
losses = FILE['LOSS']

