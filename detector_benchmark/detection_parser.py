#!/usr/bin/python3

"""
Using detections and annotations, this script does a maximal pairing between
them, and oututs a resulting .csv file that can be used to plot performance
curves.

Format of a line in the input .csv files:
"/path/to/image.jpg x1 y1 x2 y2 c x1 y1 x2 y2 c ..."
- (x1,y1): upper left corner of an object bounding box (bb).
- (x2,y2): lower right corner of bb.
- x coordinates increase from left to right, y coordinates increase from top to
  bottom.
- Width of a bb: x2-x1+1
- c: Class identifier for annotations, and detection confidence in the [0.0,1.0]
     range for detections.

Format of the output .csv file:
- recall, fp, confidence, precision

See curve_plotter.py
"""

import os
import argparse
import numpy as np
from pprint import pprint
from sys import exit
from os import path

import lib

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
    args = parseArguments()

    run(
        annotationListFile=args.annotations,
        annotationType=args.annotation_type,
        detectionListFile=args.detections,
        detectionType=args.detection_type,
        outDir=args.outdir,
        modelName=args.model_name,
        datasetName=args.dataset_name,
        iouThreshold=args.iou_threshold,
        count=args.count,
        prefix=args.prefix,
        mask=args.mask,
        delimiter=args.delimiter
    )

def run(annotationListFile,annotationType,detectionListFile,detectionType,outDir,modelName,datasetName,iouThreshold=0.5,count=1,prefix=None,mask=False,delimiter="\t"):
    annotationList = lib.readList(annotationListFile,count,delimiter=delimiter)
    detectionList = lib.readList(detectionListFile,count,detection=True,delimiter=delimiter)
    lenList = len(annotationList)
    assert lenList==len(detectionList), "annotation: {}, detection: {}".format( lenList, len(detectionList) )

    for scaleName,scaleRange in SCALES.items():
        testName = "net_"+prefix+"_" if prefix else "net_"
        testName += modelName
        testName += "_set_"
        testName += datasetName
        testName += "." + scaleName

        gtCount = 0
        detections = []
        for index in range(lenList):
            annotationLabels = annotationList[index]["labels"]
            annotationBoxes = annotationList[index]["boxes"]
            detectionScores = detectionList[index]["scores"]
            detectionBoxes = detectionList[index]["boxes"]

            assert annotationList[index]["image_path"]==detectionList[index]["image_path"], annotationList[index]["image_path"]+" != "+detectionList[index]["image_path"]

            filteredAnnotationBoxes = lib.heightFilter(annotationBoxes,minHeight=scaleRange[0],maxHeight=scaleRange[1],labels=annotationLabels if mask else None)
            positiveCount = len( filteredAnnotationBoxes )
            gtCount += positiveCount

            detection = lib.maxPair(detectionScores,detectionBoxes,detectionType,annotationLabels,annotationBoxes,annotationType,iouThreshold,minHeight=scaleRange[0],maxHeight=scaleRange[1],mask=mask)
            detections.extend(detection)

        print( "\n{}\n{}\ndetections: {}".format(testName,len(testName)*'=',len(detections)) )
        print(           "gt:         {}".format(gtCount) )

        if not gtCount:
            continue

        detections.sort(
            reverse=True,
            key=lambda row: (row[0],row[1]))

        outputPath = path.expanduser(outDir)
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

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotations", required=True)
    parser.add_argument("--annotation_type", choices=["head","body"], required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--detection_type", choices=["head","body"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)

    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--delimiter", default="\t")

    return parser.parse_args()



if __name__ == '__main__':
    main()
