#!/usr/bin/python3

import numpy as np
from pprint import pprint
from sys import exit
from os import path
import argparse, os, sys, subprocess

import util

# SCALES = {
#     "small":(8,176),
#     "medium":(176,344),
#     "large":(344,512),
#     "all":(None,None)
# }

SCALES = {
    "small": (16,96),
    "medium":(96,176),
    "large": (176,256),
    "all":   (None,None)
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="/home/bjenei/train/head_det/v3/model.desc", help="model description file")
    parser.add_argument("-l", "--list", default="/home/bjenei/list/test/hh.csv", help="list file with test examples")
    parser.add_argument("-c" ,"--count", default=1, type=int, help="input image count")
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("-o", "--output", default="~/detector_benchmark_result")

    parser.add_argument("--dldemo_exe", default="/home/bjenei/dldemo")
    parser.add_argument("--dldemo_cmd", default="rpndet")
    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--csv_path", default="/tmp/rpn_det.csv")
    parser.add_argument("--min_height", default=1, type=int)
    parser.add_argument("--max_height", default=1000, type=int)
    parser.add_argument("--min_height_pair", default=None, type=int)
    parser.add_argument("--max_height_pair", default=None, type=int)
    parser.add_argument("--confidence_threshold", default=0.01, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mask", action="store_true")

    args = parser.parse_args()

    annotationList = util.readList(args.list,args.count)

    clonedEnv = os.environ.copy()
    clonedEnv["GLOG_minloglevel"] = "2"
    ret = subprocess.run([
        args.dldemo_exe,
        args.dldemo_cmd,
        args.model,
        args.list,
        args.csv_path,
        str(args.min_height),
        str(args.max_height),
        str(args.confidence_threshold),
        str(args.count),
        str(int(args.verbose))
    ],env=clonedEnv)

    assert not ret.returncode, ret

    detectionList = util.readList(args.csv_path,args.count,detection=True)
    lenList = len(annotationList)
    assert lenList==len(detectionList), "annotation: {}, detection: {}".format( lenList, len(detectionList) )

    for scaleName,scaleRange in SCALES.items():
        testName = "net_"+args.prefix+"_" if args.prefix is not None else "net_"
        testName += path.dirname(args.model).split(os.sep)[-1]
        testName += "_set_"
        testName += path.splitext(path.basename(args.list))[0]
        testName += "." + scaleName

        gtCount = 0
        detections = []
        for index in range(lenList):
            annotationLabels = annotationList[index]["labels"]
            annotationBoxes = annotationList[index]["boxes"]
            detectionScores = detectionList[index]["scores"]
            detectionBoxes = detectionList[index]["boxes"]

            assert annotationList[index]["image_path"]==detectionList[index]["image_path"], annotationList[index]["image_path"]+" != "+detectionList[index]["image_path"]

            filteredAnnotationBoxes = util.heightFilter(annotationBoxes,minHeight=scaleRange[0],maxHeight=scaleRange[1],labels=annotationLabels if args.mask else None)
            positiveCount = len( filteredAnnotationBoxes )
            gtCount += positiveCount

            detection = util.maxPair(detectionScores,detectionBoxes,annotationLabels,annotationBoxes,args.iou_threshold,minHeight=scaleRange[0],maxHeight=scaleRange[1],mask=args.mask)
            detections.extend(detection)

        print( "\n{}\n{}\ndetections: {}".format(testName,len(testName)*'=',len(detections)) )
        print(           "gt:         {}".format(gtCount) )

        detections.sort(
            reverse=True,
            key=lambda row: (row[0],row[1]))

        outputPath = path.expanduser(args.output)
        os.makedirs(outputPath, exist_ok=True)

        tp,fp = 0,0
        lenDetections = len(detections)
        with open( path.join(outputPath, (testName+".csv")), 'w' ) as f:
            for i in range(lenDetections):
                detection = detections[i]

                if detection[1] == 1:
                    tp += 1
                else:
                    fp += 1

                # ["recall","fp","confidence","precision"]
                data = [ "{:.7f}".format(tp/gtCount), "{:d}".format(fp), "{:.7f}".format(detection[0]), "{:.7f}".format(tp/(tp+fp)) ]
                line = "\t".join(data)+"\n"
                f.write(line)

    print()



if __name__ == "__main__":
    main()
