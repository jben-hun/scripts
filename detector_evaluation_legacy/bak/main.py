#!/usr/bin/python3

import os, sys

HOME = os.environ["HOME"]
CAFFE_PATH = HOME+"/git/caffe/build"
os.environ["CAFFE_PATH"] = CAFFE_PATH
PYTHONPATH = CAFFE_PATH+"/../python"
try:
    sys.path.index(PYTHONPATH)
except ValueError:
    sys.path.append(PYTHONPATH)

import numpy as np
from pprint import pprint
from sys import exit
from os import path
from PIL import Image, ImageDraw
import argparse

import util
from rpn_detector import RpnDetector

import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="/home/bjenei/train/head_det/v3/model.desc", help="model description file")

    parser.add_argument("-l", "--list", default="/home/bjenei/list/test/test_hh.csv", help="list file with test examples")

    parser.add_argument("-n", "--name", default="test")
    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--confidence_threshold", default=0.02, type=float)
    parser.add_argument("--min_height", default=1, type=int)
    parser.add_argument("--max_height", default=1000, type=int)
    args = parser.parse_args()

    detector = RpnDetector(args.model)
    testList = util.readList(args.list)

    detections = []
    lenList = len(testList)
    gtCount = 0
    array = []

    for i,e in enumerate(testList):
        print("{}: {}: {}".format(args.name,lenList,i+1))

        scores, boxes, im = detector.infer(e["imagePath"], confidenceThreshold=args.confidence_threshold, minHeight=args.min_height, maxHeight=args.max_height)

        array.append(scores)

        # detection = util.greedyPair(scores, boxes, e["boxes"], args.iou_threshold)
        #
        # detections.extend(detection)
        #
        # gtCount += len(e["boxes"])

        # if im is not None:
        #     draw = ImageDraw.Draw(im)
        #     for box in (boxes):
        #         draw.rectangle( ((box[0], box[1]), (box[2], box[3])), outline=(0,255,0) )
        #         pass
        #     if len(scores) > 0:
        #         im.save( str(i+1) + '_' + str(min(scores))[2:] + '_' + str(max(scores))[2:] + ".jpg" )
        #     else:
        #         im.save( str(i+1) + ".jpg" )

    # detections.sort(
    #     reverse=True,
    #     key=lambda row: (row[0],row[1])
    # )
    #
    # previousScore = 1.1
    # tp,fp = 0,0
    # print( "\n{}".format(len(detections)) )
    # with open("result_"+args.name+".csv", 'w') as f:
    #     for i in range(len(detections)):
    #         k = False
    #         if detections[i][1] == 1:
    #             tp += 1
    #         else:
    #             fp += 1
    #             k = True
    #
    #         if previousScore > detections[i][0]:
    #             previousScore = detections[i][0]
    #             k = True
    #
    #         if k:
    #             line = "{}\t{}\t{}\n".format(
    #                 tp/gtCount,
    #                 fp,
    #                 detections[i][0]
    #             )
    #             f.write(line)

    debug = array
    with open("/home/bjenei/b.pkl", "wb") as f:
        pickle.dump(debug, f, pickle.HIGHEST_PROTOCOL)

    exit()

    detections.sort(
        reverse=True,
        key=lambda row: row[0]
    )

    tp,fp = 0,0
    print( "{}: detections: {}".format(args.name,len(detections)) )
    with open("result_"+args.name+".csv", 'w') as f:
        for i in range(len(detections)):
            if detections[i][1] == 1:
                tp += 1
            else:
                fp += 1

            line = "{}\t{}\t{}\n".format(
                tp/gtCount,
                fp,
                detections[i][0]
            )
            f.write(line)

if __name__ == "__main__":
    main()
