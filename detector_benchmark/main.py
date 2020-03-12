#!/usr/bin/python3

import argparse, os, subprocess
from os import path

import pair,util

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="/home/bjenei/train/head_det/v3/model.desc", help="model description file")
    parser.add_argument("-l", "--list", default="/home/bjenei/list/test/hh.csv", help="list file with test examples")
    parser.add_argument("-c" ,"--count", default=1, type=int, help="input image count")
    parser.add_argument("-p", "--prefix")
    parser.add_argument("-o", "--output", default="~/detector_benchmark_result")

    parser.add_argument("--dldemo_exe", default="/home/bjenei/dldemo")
    parser.add_argument("--dldemo_cmd", default="rpndet")
    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--csv_path", default="/home/bjenei/rpn_det.csv")
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

    pair.run(
        annotationListFile = args.list,
        annotationType = "head",
        detectionListFile = args.csv_path,
        detectionType = "body",
        outputDir = args.output,
        modelName = path.dirname(args.model).split(os.sep)[-1],
        datasetName = path.splitext(path.basename(args.list))[0],
        iouThreshold = args.iou_threshold,
        count = args.count,
        prefix = args.prefix,
        mask = args.mask
    )



if __name__ == "__main__":
    main()
