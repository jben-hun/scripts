#!/usr/bin/python3

"""
Runs detection with a Region Proposal Network, and does the pairing necessary
for plotting performance curves.

See curve_plotter.py and detection_parser.py
"""

import argparse
import os
import subprocess
from os import path

import detection_parser

def main():
    args = parseArguments()

    clonedEnv = os.environ.copy()
    clonedEnv["GLOG_minloglevel"] = "2"
    ret = subprocess.run([
        args.dldemo_exe,
        args.dldemo_cmd,
        args.model,
        args.annotations,
        args.tempfile,
        str(args.min_height),
        str(args.max_height),
        str(args.confidence_threshold),
        str(args.count),
        "0"
    ],env=clonedEnv)

    assert not ret.returncode, ret

    detection_parser.run(
        annotationListFile=args.annotations,
        annotationType=args.annotation_type,
        detectionListFile=args.tempfile,
        detectionType=args.detection_type,
        outDir=args.outdir,
        modelName=path.dirname(args.model).split(os.sep)[-1],
        datasetName=path.splitext(path.basename(args.annotations))[0],
        iouThreshold=args.iou_threshold,
        count=args.count,
        prefix=args.prefix,
        maskOutliers=args.mask_outliers,
        delimiter=args.delimiter
    )

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="model description file", required=True)
    parser.add_argument("--annotations", help="list file with test examples", required=True)

    parser.add_argument("--annotation_type", choices=["head","body"], default="head")
    parser.add_argument("--tempfile", default="/tmp/head_detector_benchmark_temp.csv", help="where to write immediate detections with confidences")
    parser.add_argument("--detection_type", choices=["head","body"], default="head")
    parser.add_argument("--outdir", default="~/detector_benchmark_result")
    parser.add_argument("--prefix")
    parser.add_argument("--confidence_threshold", default=0.01, type=float, help="for detection, should be as low as feasible")
    parser.add_argument("--iou_threshold", default=0.5, type=float, help="for matching detections to annotations")
    parser.add_argument("--count", default=1, type=int, help="input image count")
    parser.add_argument("--mask_outliers", action="store_true", help="exclude upper outlying objects")
    parser.add_argument("--dldemo_exe", default="/home/bjenei/dldemo")
    parser.add_argument("--dldemo_cmd", default="rpndet")
    parser.add_argument("--min_height", default=1, type=int)
    parser.add_argument("--max_height", default=1000, type=int)
    parser.add_argument("--delimiter",default="\t")

    return parser.parse_args()



if __name__ == "__main__":
    main()
