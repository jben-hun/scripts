#!/usr/bin/python3

import numpy as np
from pprint import pprint
from sys import exit
from os import path
import os, argparse

from . import util

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

    parser.add_argument("--annotations", required=True)
    parser.add_argument("--annotation_type", choices=["head","body"], required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--detection_type", choices=["head","body"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--iou_threshold", type=float, required=True)

    parser.add_argument("-c" ,"--count", default=1, type=int)
    parser.add_argument("-p", "--prefix")
    parser.add_argument("-m", "--mask", action="store_true")

    args = parser.parse_args()

    run(
        annotationListFile = args.annotations,
        annotationType = args.annotation_type,
        detectionListFile = args.detections,
        detectionType = args.detection_type,
        outputDir = args.output_dir,
        modelName = args.model_name,
        datasetName = args.dataset_name,
        iouThreshold = args.iou_threshold,
        count = args.count,
        prefix = args.prefix,
        mask = args.mask
    )


def extract_detections(annotationList, annotationType, detectionList, detectionType, scaleRange, iouThreshold,
                       mask=False):
    gtCount = 0
    detections = []
    for index in range(len(annotationList)):
        annotationLabels = annotationList[index]["labels"]
        annotationBoxes = annotationList[index]["boxes"]
        detectionScores = detectionList[index]["scores"]
        detectionBoxes = detectionList[index]["boxes"]

        assert annotationList[index]["image_path"] == detectionList[index]["image_path"], \
            "{} != {}".format(annotationList[index]["image_path"], detectionList[index]["image_path"])

        filteredAnnotationBoxes = util.heightFilter(annotationBoxes, minHeight=scaleRange[0], maxHeight=scaleRange[1],
                                                    labels=annotationLabels if mask else None)
        positiveCount = len(filteredAnnotationBoxes)
        gtCount += positiveCount

        detection = util.maxPair(detectionScores, detectionBoxes, detectionType, annotationLabels, annotationBoxes,
                                 annotationType, iouThreshold, minHeight=scaleRange[0], maxHeight=scaleRange[1],
                                 mask=mask)
        detections.extend(detection)
    return detections, gtCount


def eval_scales(annotationList, annotationType, detectionList, detectionType, iouThreshold, testNamePrefix="",
                mask=False, scales=None, extract_detections_fun=extract_detections):
    if scales is None:
        scales = SCALES

    statistics = {}
    for scaleName, scaleRange in scales.items():
        detections, gtCount = extract_detections_fun(annotationList, annotationType, detectionList, detectionType,
                                                     scaleRange, iouThreshold, mask)

        if testNamePrefix:
            testName = "{}.{}".format(testNamePrefix, scaleName)
            print("\n{}\n{}\ndetections: {}".format(testName, len(testName) * '=', len(detections)))
            print("gt:         {}".format(gtCount))
            print()

        detections.sort(
            reverse=True,
            key=lambda row: (row[0], row[1])
        )

        tp, fp = 0, 0
        range_stats = []
        for detection in detections:
            if detection[1] == 1:
                tp += 1
            else:
                fp += 1

            range_stats.append((
                tp / gtCount,
                fp,
                detection[0],
                tp / (tp + fp),
            ))
        statistics[scaleName] = range_stats
    return statistics


def run(annotationListFile,annotationType,detectionListFile,detectionType,outputDir,modelName,datasetName,iouThreshold,count=1,prefix="",mask=False):
    annotationList = util.readList(annotationListFile,count)
    detectionList = util.readList(detectionListFile,count,detection=True)
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

            filteredAnnotationBoxes = util.heightFilter(annotationBoxes,minHeight=scaleRange[0],maxHeight=scaleRange[1],labels=annotationLabels if mask else None)
            positiveCount = len( filteredAnnotationBoxes )
            gtCount += positiveCount

            detection = util.maxPair(detectionScores,detectionBoxes,detectionType,annotationLabels,annotationBoxes,annotationType,iouThreshold,minHeight=scaleRange[0],maxHeight=scaleRange[1],mask=mask)
            detections.extend(detection)

        print( "\n{}\n{}\ndetections: {}".format(testName,len(testName)*'=',len(detections)) )
        print(           "gt:         {}".format(gtCount) )

        detections.sort(
            reverse=True,
            key=lambda row: (row[0],row[1]))

        outputPath = path.expanduser(outputDir)
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



if __name__ == '__main__':
    main()
