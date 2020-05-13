#!/usr/bin/python3

import argparse
import os
import sys

HOME = os.environ["HOME"]
CAFFE_PATH = HOME+"/git/caffe/build"
os.environ["CAFFE_PATH"] = CAFFE_PATH
PYTHONPATH = CAFFE_PATH+"/../python"
try:
    sys.path.index(PYTHONPATH)
except ValueError:
    sys.path.append(PYTHONPATH)

import numpy as np
from munkres import Munkres
import collections
#from math import exp

import caffe
import xml.etree.ElementTree
import cv2

from sys import exit
from pprint import pprint

caffe.set_random_seed(666)
import numpy.random
numpy.random.seed(666)
import random
random.seed(666)
cv2.setRNGSeed(666)

def IOU_rects(r1, r2):
    if r1[2] < r2[0] or r2[2] < r1[0] or r1[3] < r2[1] or r2[3] < r1[1]:
        return 0

    SI = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0])) * max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    S1 = (r1[2]-r1[0])*(r1[3]-r1[1])
    S2 = (r2[2]-r2[0])*(r2[3]-r2[1])
    S = S1+S2-SI

    return SI/S if S != 0 else 0


def nms(dets, thresh):
    if not dets.shape[0]:
      return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def calcWiseMatrix(annots, dets):
    wise = []
    for annot in annots:
        wise_row = []
        for det in dets:
            wise_row.append(1-IOU_rects(annot, det))
        wise.append(wise_row)
    return wise


def vis_detections_opencv(im, class_name, dets, color ,thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), font, 1, color, 1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='path_net', help='Path of Proto', default="/home/bjenei/train/head_det/v3/deploy.prototxt")
    parser.add_argument('--model', dest='path_model', help='Path of .caffemodel', default="/home/bjenei/train/head_det/v3/net.caffemodel")
    parser.add_argument('--list',dest='list', default="/home/bjenei/list/test/test_hh.csv")
    parser.add_argument("-n", "--name", default="test")
    args = parser.parse_args()

    return args


def readList(listFile):
    testList = []
    with open(listFile, 'r') as f:
        lines = f.readlines()

    lines.sort()
    for line in lines:
        line = line.rstrip()
        if line == '' or line[0] == '#':
            continue

        line = line.replace(' ','')
        lineSplit = line.split('\t')

        boxes_ = [ int(float(lineSplit[i])) for i in range(1,len(lineSplit)) if i%5!=0 ]
        boxes = np.reshape( np.array(boxes_,dtype=np.int32), (-1,4) )

        testList.append({
            "imagePath":lineSplit[0],
            "boxes":boxes
        })

    return testList


def im_dete(net, frame, threshold, minHeight, maxHeight, frameName):
    im_orig = frame.astype(np.float32, copy=True)

    im_orig -= [104,117,123]

    pad_h = 32 - im_orig.shape[0] % 32
    pad_h =  0 if pad_h is 32 else pad_h

    pad_w = 32 - im_orig.shape[1] % 32
    pad_w =  0 if pad_w is 32 else pad_w

    blob = np.zeros((1, im_orig.shape[0]+pad_h, im_orig.shape[1]+pad_w, 3), dtype=np.float32)

    blob[0, 0:im_orig.shape[0], 0:im_orig.shape[1], :] = im_orig

    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    net.blobs['data'].reshape(*(blob.shape))

    im_input2 = np.empty(shape=blob.shape,dtype=blob.dtype)
    im_input2[...] = blob[...]
    blob = im_input2
    forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
    net.forward(**forward_kwargs)

    blob_scores = net.blobs['rpn_cls_prob_reshape'].data.copy()
    blob_boxes = net.blobs['rpn_bbox_pred1'].data.copy()

    return blob_scores, blob_boxes

    boxes = []
    scores= []
    scales = [1,2,4,8,16]
    for i in range(0, len(scales)):
        for y in range(0,blob_scores.shape[2]):
            for x in range(0, blob_scores.shape[3]):
                if blob_scores[0,len(scales)+i,y,x] > threshold:
                    size = scales[i]*16
                    x_corr = blob_boxes[0,4*i,y,x] * size
                    y_corr = blob_boxes[0,4*i+1,y,x] * size
                    w_pred = np.exp(blob_boxes[0,4*i+2,y,x]) * size
                    h_pred = np.exp(blob_boxes[0,4*i+3,y,x]) * size

                    center_x = x*16+8 + x_corr
                    center_y = y*16+8 + y_corr

                    x1 = center_x-w_pred/2
                    y1 = center_y-h_pred/2
                    x2 = center_x+w_pred/2
                    y2 = center_y+h_pred/2

                    if ( x1 >= 0 and y1 >= 0 and x2 <= im_orig.shape[1] and y2 <= im_orig.shape[0] and y2 - y1 + 1 >= minHeight and y2 - y1 + 1 <= maxHeight ):
                        boxes.append([x1,y1,x2,y2])
                        scores.append(blob_scores[0,len(scales)+i,y,x])

    return scores, boxes

if __name__ == '__main__':
    CONF_THRESH = 0.02
    NMS_THRESH = 0.3
    MIN_HEIGHT = 1
    MAX_HEIGHT = 1000
    array = []

    args = parse_args()


    if not os.path.isfile(args.path_model):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(args.path_model))


    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.path_net, args.path_model, caffe.TEST)

    imdb = os.path.join(args.list)

    full_annot_num = 0
    full_det_list=[]

    matcher = Munkres()

    testList = readList(args.list)

    pic_num=0

    for row in testList:
        pic_num+=1
        print( pic_num)
        frame_path = row["imagePath"]
        gt_boxes = row["boxes"]

        if len(gt_boxes)==0:
            continue

        frame = cv2.imread(frame_path)

        scores, boxes = im_dete(net, frame, CONF_THRESH, MIN_HEIGHT, MAX_HEIGHT, frame_path)

        array.append(scores)
        continue

        #get bboxes of second class

        cls_boxes  = np.array(boxes)
        cls_scores = np.array(scores)

        dets = np.empty([0,5])
        if cls_boxes.shape[0]:
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float64)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        delete_num=0
        for ind in np.where(dets[:, -1] < CONF_THRESH)[0]:
            dets=np.delete(dets,(ind-delete_num),axis=0)
            delete_num+=1

        full_annot_num+=len(gt_boxes)

        if len(dets)==0:
            continue

        wise = calcWiseMatrix(gt_boxes, dets)

        # indexes = matcher.compute(wise)
        indexes = []
        used = set()
        for i in range(len(gt_boxes)):
            minVal = 0
            minInd = -1
            for j in range(len(dets)):
                if (j not in used) and (wise[i][j] < minVal or minInd == -1):
                    minVal = wise[i][j];
                    minInd = j;
            if (minInd != -1):
                used.add(minInd);
                indexes.append((i,minInd))

        det_list = []

        for i in range(0, len(dets)):
            b = [item for item in indexes if item[1] == i]
            if len(b) == 1 and wise[b[0][0]][b[0][1]] <= 0.5:
                det_list.append((dets[i][4],True))
            else:
                det_list.append((dets[i][4],False))

        full_det_list.extend(det_list)

    debug = array
    np.save("/home/bjenei/j.npy",debug)
    exit()

    full_det_list.sort( key=lambda tup: tup[0], reverse=True )

    tp,fp = 0,0
    with open("faster_"+args.name+".csv", 'w') as f:
        for i in range(len(full_det_list)):
            if full_det_list[i][1] == 1:
                tp += 1
            else:
                fp += 1

            line = "{}\t{}\t{}\n".format(
                tp/full_annot_num,
                fp,
                full_det_list[i][0]
            )
            f.write(line)
