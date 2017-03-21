import mlp
import numpy as np
#import pylab as pl
import sys

def preprocessFacebook(fileIn, fileOut):
    Photo = '0'
    Status = '1'
    Link = '2'
    Video = '3'
    with open(fileIn, 'r') as fin:
        outF = open(fileOut, 'w')
        for data in fin:
            if data.find(';;') == -1:
                if data.find('Photo') > -1:
                    outF.write(data.replace('Photo', Photo).replace('\r', ''))
                elif data.find('Status') > -1:
                    outF.write(data.replace('Status', Status).replace('\r', ''))
                elif data.find('Link') > -1:
                    outF.write(data.replace('Link', Link).replace('\r', ''))
                elif data.find('Video') > -1:
                    outF.write(data.replace('Video', Video).replace('\r', ''))

        outF.close()



def main():
    preprocessFacebook('dataset_Facebook.csv', 'facebook.csv')
    data = np.loadtxt('facebook.csv', delimiter = ';')

    data[:,2:] = data[:,2:]-data[:,2:].mean(axis = 0)
    imax = np.concatenate((data.max(axis=0) * np.ones((1,19)), np.abs(data.min(axis=0)) * np.ones((1,19))), axis=0).max(axis=0)
    data[:,2:] = data[:,2:]/imax[2:]
    data[:,:1] = data[:,:1]-data[:,:1].mean(axis = 0)
    data[:,:1] = data[:,:1]/imax[:1]
    trainSet = data[:250,0:18]
    trainTarget = data[:250]
    testSet = data[251:374,0:18]
    testTarget = data[251:374]
    validSet = data[375:500,0:18]
    validTarget = data[375:500]

    net = mlp.mlp(trainSet, trainTarget, 30, outtype = 'linear')
    net.earlystopping(trainSet, trainTarget, validSet, validTarget, 0.4, 5000)
    net.confmat(testSet,testTarget)




if __name__ == '__main__':
    main()
