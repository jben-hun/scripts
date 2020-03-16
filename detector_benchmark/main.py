#!/usr/bin/python3

"""
Runs a detection with a Region Proposal Network, and does the pairing necessary
for plotting perrformance curves.
See roc.py and detector_benchmark/pair.py
"""

import argparse
import os
import subprocess
from os import path

import pair
import util

def main():
    args = parseArguments()

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

    pair.run(
        annotationListFile=args.list,
        annotationType=args.annotation_type,
        detectionListFile=args.csv_path,
        detectionType=args.detection_type,
        outputDir=args.output,
        modelName=path.dirname(args.model).split(os.sep)[-1],
        datasetName=path.splitext(path.basename(args.list))[0],
        iouThreshold=args.iou_threshold,
        count=args.count,
        prefix=args.prefix,
        mask=args.mask,
        delimiter=args.delimiter
    )

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="/home/bjenei/train/head_det/v3/model.desc", help="model description file")
    parser.add_argument("-l", "--list", default="/home/bjenei/list/test/hh.csv", help="list file with test examples")
    parser.add_argument("-c" ,"--count", default=1, type=int, help="input image count")
    parser.add_argument("-p", "--prefix")
    parser.add_argument("-o", "--output", default="~/detector_benchmark_result")
    parser.add_argument("--dldemo_exe", default="/home/bjenei/dldemo")
    parser.add_argument("--dldemo_cmd", default="rpndet")
    parser.add_argument("--iou_threshold", default=0.5, type=float, help="for matching detections to annotations")
    parser.add_argument("--csv_path", default="/tmp/rpn_det.csv", help="where to write immediate detections with confidences")
    parser.add_argument("--min_height", default=1, type=int)
    parser.add_argument("--max_height", default=1000, type=int)
    parser.add_argument("--min_height_pair", default=None, type=int)
    parser.add_argument("--max_height_pair", default=None, type=int)
    parser.add_argument("--confidence_threshold", default=0.01, type=float, help="for detection, should be as low as feasible")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--annotation_type", choices=["head","body"], default="head")
    parser.add_argument("--detection_type", choices=["head","body"], default="head")
    parser.add_argument("-d", "--delimiter",default="\t")

    return parser.parse_args()



if __name__ == "__main__":
    main()
