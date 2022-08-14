#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt


# In[2]:


d_model = 100
d_hid = 128 # 原本512
n_layer = 3
n_head = 4


# In[3]:


vocab,embeddings = [],[]
with open('glove.6B.100d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)
#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))


# In[4]:


def src_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
MASK = src_mask(156)
#d_model = 128
class Classification2(nn.Module):
    def __init__(self):
        super(Classification2, self).__init__()
        TEMP = nn.TransformerEncoderLayer(d_model = d_model, 
                                          nhead = n_head, 
                                          dropout = 0.3, 
                                          dim_feedforward = d_hid, 
                                          activation = 'relu',  
                                          batch_first = True)
        self.encoder = nn.TransformerEncoder(TEMP, num_layers = n_layer)
        self.Dense = nn.Linear(15600, 4)
    def forward(self, enc_input):
        x = self.encoder(enc_input, MASK)
        x = x.reshape(x.shape[0], -1) # (N, 15600)
        x = nn.Tanh()(x)
        x = self.Dense(x)
        predict = nn.Softmax(dim = 1)(x)
        return predict


# In[5]:


model = Classification2()
model.load_state_dict(torch.load('309707001_transformer.pt'))


# In[6]:


import torch
import torch.nn as nn
def data_setting(sentences1, en_dict):
    enc_input = list()
    N = len(sentences1)
    for idx in range(N):
        n = len(sentences1[idx])
        alist = list()
        for i in range(156):
            if i < n:
                word = sentences1[idx][i]
                if word not in en_dict:
                    alist.append(1)
                else:
                    alist.append(en_dict.index(word))
            else:
                alist.append(0)
        enc_input.append(alist)
        print("\r完成進度{0}".format((idx + 1)/N), end='')
    return torch.LongTensor(enc_input)


# In[7]:


Test_File = pd.read_csv('./news_data/test.csv')
test_sent = list()
for idx in range(Test_File.shape[0]):
    temp1 = Test_File['Title'][idx] + '. ' + Test_File['Description'][idx]
    temp1 = temp1.replace('\\', ' ')
    temp1 = temp1.replace('-', ' ')
    while True:
        if 'http' in temp1:
            I = temp1.find('&lt')
            II = temp1.find('A&gt')
            unwant = temp1[I:II + 4]
            temp1 = temp1.replace(unwant, ' ')
        else:
            break
    line = temp1
    line = line.lower()
    temp = word_tokenize(line)
    alist = [word for word in temp]
    test_sent.append(alist)

test_input = data_setting(test_sent, list(vocab_npa))


# In[8]:


test_in = torch.tensor(embs_npa[test_input], dtype = torch.float32)


# In[9]:


model.eval()
with torch.no_grad():
    Y = model(test_in)
    torch.manual_seed(0)
    RESULT = pd.DataFrame(np.array(torch.argmax(Y, axis = 1)), columns = ['Category']).reset_index()
    RESULT.columns = ['Id','Category']
    RESULT['Id'] = Test_File['Id']
    RESULT['Category'] += 1
    RESULT.to_csv('309707001_submission.csv', index = False)


# In[ ]:





# In[ ]:




