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

    data = []
    with open('facebook.csv', 'r') as f:
        for i in f:
            data.append(i.replace('\n', '').split(';'))

    print data[::3,:]
    #data = np.loadtxt('facebook.csv', delimiter = ';')

    trainSet = data[:250]
    testSet = data[250:375]
    validSet = data[375:500]

    net = mlp.mlp(trainSet, trainSet, 3, outtype = 'linear')
    net.mlptrain(trainSet, trainSet, 0.4, 500)




if __name__ == '__main__':
    main()
