#!/usr/bin/python3

import numpy as np
from pprint import pprint
from sys import exit
from os import path
from decimal import getcontext, Decimal
import argparse, os, sys, subprocess

import util

def main():
    getcontext().prec = 6

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="/home/bjenei/train/head_det/v3/model.desc", help="model description file")

    parser.add_argument("-l", "--list", default="/home/bjenei/list/test/hh.csv", help="list file with test examples")

    parser.add_argument("-c" ,"--count", default=1, type=int, help="input image count")

    parser.add_argument("-p", "--prefix", default=None)

    parser.add_argument("-o", "--output", default="~/detector_benchmark_result")

    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--csv_path", default="/tmp/rpn_det.csv")
    parser.add_argument("--min_height", default=1, type=int)
    parser.add_argument("--max_height", default=1000, type=int)
    parser.add_argument("--confidence_threshold", default=0.02, type=float)
    parser.add_argument("--min_height_pair", default=None, type=int)
    parser.add_argument("--max_height_pair", default=None, type=int)

    args = parser.parse_args()

    annotationList = util.readList(args.list,args.count)

    ret = subprocess.run([
        "/home/bjenei/dldemo",
        "rpndet",
        args.model,
        args.list,
        args.csv_path,
        str(args.min_height),
        str(args.max_height),
        str(args.confidence_threshold),
        str(args.count),
        '0'
    ])
    assert not ret.returncode, ret.returncode

    detectionList = util.readList(args.csv_path,args.count,detection=True)

    lenList = len(annotationList)

    assert lenList==len(detectionList), "annotation: {}, detection: {}".format( lenList, len(detectionList) )

    if args.prefix is not None:
        testName = args.prefix + "_"
    else:
        testName = ""

    testName += path.dirname(args.model).split(os.sep)[-1]
    testName += "_"
    testName += path.splitext(path.basename(args.list))[0]

    gtCount = 0
    detections = []
    for index in range(lenList):
        #print("{}: {}: {}".format(testName,lenList,index+1))

        annotationBoxes = annotationList[index]["boxes"]
        detectionScores = detectionList[index]["scores"]
        detectionBoxes = detectionList[index]["boxes"]

        assert annotationList[index]["image_path"]==detectionList[index]["image_path"], annotationList[index]["image_path"]+" != "+detectionList[index]["image_path"]

        gtCount += len( util.heightFilter(annotationBoxes,minHeight=args.min_height_pair,maxHeight=args.max_height_pair) )

        detection = util.maxPair(detectionScores,detectionBoxes,annotationBoxes,args.iou_threshold,minHeight=args.min_height_pair,maxHeight=args.max_height_pair)
        detections.extend(detection)

    detections.sort(
        reverse=True,
        key=lambda row: (row[0],row[1])
    )

    print( "\n{}: detections: {}".format(testName,len(detections)) )
    print( "gt: {}\n".format(gtCount) )

    outputPath = path.expanduser(args.output)
    os.makedirs(outputPath, exist_ok=True)

    tCount = gtCount + np.sum( 1 - (np.array(detections)[:,1]) )

    tp,fp = 0,0
    lenDetections = len(detections)
    with open( path.join(outputPath, (testName+".csv")), 'w' ) as f:
        for j in range(1, lenDetections+1):
            last = (j == lenDetections)
            i = j - 1
            detection = detections[i]

            if detection[1] == 1:
                tp += 1
            else:
                fp += 1

            if last or not detections[j][1]:
                data = ( 1*Decimal(tp/gtCount), fp, 1*Decimal(fp/tCount), 1*Decimal(detection[0]) )
                line = "{}\t{}\t{}\t{}\n".format(*data)
                f.write(line)


if __name__ == "__main__":
    main()
