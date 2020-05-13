#!/usr/bin/python3

import os
import sys
import numpy as np
import argparse

HOME = os.environ["HOME"]
CAFFE_PATH = HOME+"/git/caffe/build"
os.environ["CAFFE_PATH"] = CAFFE_PATH
PYTHONPATH = CAFFE_PATH+"/../python"
try:
    sys.path.index(PYTHONPATH)
except ValueError:
    sys.path.append(PYTHONPATH)

import caffe


parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("target")
args = parser.parse_args()

os.chdir(args.source)
oldnet = caffe.Net('train.prototxt', caffe.TRAIN, weights='lite.caffemodel')
a = oldnet.params['conv1'][0].data
b = oldnet.params['conv1'][1].data

os.chdir(args.target)
net = caffe.Net('train.prototxt', caffe.TRAIN, weights='../v3/lite.caffemodel')
net.params['conv1b'][0].data[:, :3, ...] = (a/2)
net.params['conv1b'][0].data[:, 3:, ...] = (a/2)
net.params['conv1b'][1].data[:] = b
net.save('lite.caffemodel')

net = caffe.Net('train.prototxt', caffe.TRAIN, weights='lite.caffemodel')
a = net.params['conv1b'][0].data
b = net.params['conv1b'][1].data

print(
    np.sum(np.absolute(a[:, :3, ...] - a[:, 3:, ...])),
    np.sum(np.absolute(a)),
    np.sum(np.absolute(b)),
    sep="\n\n"
)
