import mlp
import numpy as np
import pylab as pl
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
    data = data-data.mean(axis = 0)
    imax = np.concatenate((data.max(axis=0)*np.ones((1,19)),np.abs(data.min(axis=0)*np.ones((1,19)))),axis=0).max(axis=0)
    data = data/imax

    train = data[::2,:]
    trainTarget = data[::2,18].reshape((np.shape(train)[0]),1)
    valid = data[1::4,:]
    validTarget = data[1::4,18].reshape((np.shape(valid)[0]),1)
    test = data[3::4,:]
    testTarget = data[3::4,18].reshape((np.shape(test)[0]),1)

    net = mlp.mlp(train, trainTarget, 30, outtype = 'linear')
    net.earlystopping(train, trainTarget, valid, validTarget, 0.4, 500)

    test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
    testout = net.mlpfwd(test)

    pl.figure()
    pl.plot(np.arange(np.shape(test)[0]),testout,'.')
    pl.plot(np.arange(np.shape(test)[0]),testTarget,'x')
    pl.legend(('Predictions','Targets'))
    pl.show()

if __name__ == '__main__':
    main()
