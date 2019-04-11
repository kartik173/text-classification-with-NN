# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:29:57 2019

@author: kartik
"""
'''
import pandas as pd
import json
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 


def get_filter_words(example_sent):
  
    stop_words = set(stopwords.words('english'))
    
    example_sent=example_sent.lower()
    word_tokens = word_tokenize(example_sent)
    
    filtered_words = [] 
    for i in word_tokens:
        if i.isalpha():
            filtered_words.append(i)
            
     
    filtered_words1 = []
    for w in filtered_words: 
        if w not in stop_words: 
            filtered_words1.append(w) 
      
    filtered_words1=set(filtered_words1)
    
    ps=PorterStemmer()
    
    temp=[]
    for i in filtered_words1:
        temp.append(ps.stem(i))
        
    return temp


depart_doc=pd.read_csv('data/document_departments.csv')

li=[]
m=depart_doc["Document ID"]
#print(len(m))
for i in range(len(m)):
    #print(i)
    st='data/docs/'+str(m[i])+".json"
    li.append([depart_doc["Department"][i]])
    
    with open(st) as f:
        dat = json.load(f)
        des=dat["jd_information"]["description"]
        if des=="":
            des=pd.NaT
        li[i].append(des)
    
    
data=pd.DataFrame(columns=["ID","Description"],data=li)

data.count()

dataset=data.dropna()

dataset.groupby("ID").count()

columns=set([])
rows=[]
dictn={}

for i in dataset["Description"]:
    t=get_filter_words(i)
    for wrd in t:
        columns.add(wrd)
    rows.append(t)
    
for i in columns:
    dictn[i]=[]
    for j in rows:
        dictn[i].append(j.count(i))
        
        
df=pd.DataFrame(data=dictn)
df["Category"],_=pd.factorize(dataset["ID"])

df.to_csv(path_or_buf="spot_mod_data.csv",index=False)

'''

import pandas as pd

dataset=pd.read_csv('data/spot_mod_data.csv')
q1=dataset.iloc[:,:-1].sum(axis=0)
q2=list(q1[q1<4].index)
ds=dataset.drop(columns=q2)
#x=ds.iloc[:,:-1]/ds.iloc[:,:-1].sum(axis=0)
#print(ds.size())

print(ds[:10])
i=int(0.2* len(ds))
j=int(0.2*i)
ds[j:-i].to_csv(path_or_buf="train.csv",index=False)
ds[:j].to_csv(path_or_buf="validate.csv",index=False)
ds[-i:].to_csv(path_or_buf="test.csv",index=False)

'''
zz=pd.read_csv('test.csv').values

print(int(0.2* len(ds)))


xtrain=ds.iloc[:-i,:].values
xtest=ds.iloc[-i:,:].values
ytrain=ds.iloc[:-i,-1].values
ytest=ds.iloc[-i:,-1].values

print(ytest)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
'''

import torch
from torch import nn
#from torchvision import datasets
import torch.utils.data as data
from torch import optim
import numpy as np

class my_points(data.Dataset):
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values   # Read data file.
        self.data = pd_data[:,:-1]   # 1st and 2nd columns --> x,y
        self.target = pd_data[:,-1:]  # 3nd column --> label
        self.n_samples = self.data.shape[0]
    
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])
# We create the dataloader.
train= my_points('train.csv')
test=my_points('test.csv')
validate=my_points('validate.csv')

batch_size = 20
trainloader = data.DataLoader(train,batch_size=batch_size,num_workers=0)
testloader = data.DataLoader(test,batch_size=batch_size,num_workers=0)
validloader = data.DataLoader(validate,batch_size=batch_size,num_workers=0)


# define a model
model = nn.Sequential(nn.Linear(1203, 512),
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(256, 27),
                      nn.LogSoftmax(dim=1)
                      )

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)


n_epochs=1000
valid_loss_min = np.Inf # set initial "min" to infinity

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for datas, target in trainloader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(datas)
        target = target.long()
        target=target.squeeze(1)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*datas.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval() # prep model for evaluation
    for data1, target in validloader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data1)
        # calculate the loss
        target = target.long()
        target=target.squeeze(1)
        loss = criterion(output, target)
        # update running validation loss 
        valid_loss += loss.item()*data1.size(0)
        
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(trainloader.dataset)
    valid_loss = valid_loss/len(validloader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model, 's_model.pt')
        valid_loss_min = valid_loss








'''
# train the model
epochs = 50
for e in range(epochs):
    running_loss = 0
    for datas, target in trainloader:
        target = target.long()
        target=target.squeeze(1)
        optimizer.zero_grad()
        
        output = model(datas)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(running_loss,f"Training loss: {running_loss/len(trainloader)}")
        
'''

model=torch.load('s_model.pt')
# test the model

test_loss=0.0

matched=[0 for i in range(27)]
total=[0 for i in range(27)]
with torch.no_grad():
    model.eval()
    for dat,target in testloader:
        
        output=model(dat)
        #print(output)
        _,pred=torch.max(output,1)
        target = target.long()
        target=target.squeeze(1)
        loss=criterion(output,target)
        test_loss+=loss.item()*dat.size(0)
        #correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        
        for i in range(len(pred)):
            xx=int(target[i].item())
            if xx==pred[i]:
                matched[xx]+=1
            total[xx]+=1
            
    test_loss=test_loss/len(test)

print("test loss",test_loss)
#for i in range(len(matched)):
#    print("Accuracy of",alphabet[i],"is:",int((matched[i]/total[i])*100), str(matched[i])+"/"+str(total[i]))
    
print("overall accuracy:",(sum(matched)/sum(total))*100)