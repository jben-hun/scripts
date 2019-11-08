#!/usr/bin/python3

import os, sys
import numpy as np
np.set_printoptions(suppress=True)

HOME = os.environ["HOME"]
CAFFE_PATH = HOME+"/git/caffe/build"
os.environ["CAFFE_PATH"] = CAFFE_PATH
PYTHONPATH = CAFFE_PATH+"/../python"
try:
    sys.path.index(PYTHONPATH)
except ValueError:
    sys.path.append(PYTHONPATH)

import caffe

def getDistance(a_):
    a = a_.reshape( (a_.shape[0],-1) )
    a = np.sqrt( np.sum(a**2,axis=1) )

    return a

os.chdir("/home/bjenei/train/head_det/tagged_multi2")
net = caffe.Net('train.prototxt', caffe.TRAIN, weights='net.caffemodel')
a = net.params['conv1b'][0].data
old = a[:,:3,...]
new = a[:,3:,...]

os.chdir("../tagged_multi1")
net = caffe.Net('train.prototxt', caffe.TRAIN, weights='net.caffemodel')
single = net.params['conv1'][0].data

net = caffe.Net('train.prototxt', caffe.TRAIN, weights='lite.caffemodel')
lite = net.params['conv1'][0].data

oldNorm = getDistance(old)
newNorm = getDistance(new)
singleNorm = getDistance(single)
liteNorm = getDistance(lite)
norms = np.hstack((oldNorm[:,np.newaxis],newNorm[:,np.newaxis],singleNorm[:,np.newaxis],liteNorm[:,np.newaxis]))
print("\nframe1 | frame2(annotated) | single-input-frame | imagenet pretrain")
print(norms)
print("\nmeans")
print( norms.mean(axis=0,dtype=np.float64) )
print("\nmean difference (frame2-frame1)")
print( np.mean(norms[:,1]-norms[:,0]) )
