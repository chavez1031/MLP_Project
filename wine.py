import numpy as np
import mlp

#import pylab as pl

import os

wine = np.loadtxt('wine.data',delimiter=',')


#normolizing
wine[:,1:] = wine[:,1:]-wine[:,1:].mean(axis=0)

imax = np.concatenate((wine.max(axis=0)*np.ones((1,14)),np.abs(wine.min(axis=0)*np.ones((1,14)))),axis=0).max(axis=0)    #find max of each dimension
wine[:,1:] = wine[:,1:]/imax[1:]
#print wine[0:5,:]



#set target
target = np.zeros((np.shape(wine)[0],3));   #3 is that we have 3 classes of wine.
indices = np.where(wine[:,0]==1)
target[indices,0] = 1
indices = np.where(wine[:,0]==2)
target[indices,1] = 1
indices = np.where(wine[:,0]==3)
target[indices,2] = 1

# Randomly order the data
order = range(np.shape(wine)[0])
np.random.shuffle(order)
wine = wine[order,:]
target = target[order,:]

# Split into training, validation, and test sets

train = wine[::2,1:]    #train set = 50%
traint = target[::2]    #traintarget= 50%
valid = wine[1::4,1:]   #valid=25%
validt = target[1::4]
test = wine[3::4,1:]
testt = target[3::4]

#print train.max(axis=0), train.min(axis=0)

# Train the network

net = mlp.mlp(train,traint,13,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)
