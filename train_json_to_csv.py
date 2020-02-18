#! /usr/bin/python3

import json, os, csv, argparse, scipy.optimize
import os.path as path
from pprint import pprint
from datetime import datetime
import sys, glob
from math import ceil, floor
from sys import exit
import numpy as np

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
IOU_THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-c","--count",default=1,type=int,help="input image count")
    parser.add_argument("-i","--ignore",nargs='+',default=["head_det","headdet","head det"],help="annotator(s) to ignore")
    parser.add_argument("-a","--annotator",action="store_true",help="output annotator names")
    parser.add_argument("-d","--date_limit",help="upper bound for annotation date")
    parser.add_argument("-m","--merge",action="store_true",help="try to clean up annotations by merging them and dropping unmerged ones")
    args = parser.parse_args()

    name = args.path.replace(os.sep,"")
    dateLimit =  None if args.date_limit is None else datetime.strptime(args.date_limit, DATE_FORMAT)

    args.annotator = args.annotator and not args.merge

    if args.annotator:
        csvfileAnnotator = open( name + "_annotator.csv", 'w', newline='' )
        csvwriterAnnotator = csv.writer(csvfileAnnotator, delimiter="\t", lineterminator="\n")

    with open( name + ".csv", 'w', newline='' ) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
        c = 0
        for (dir, dirs, filenames) in os.walk(args.path):
            for file in filenames:
                if ".json" in file and ".log.json" not in file:
                    c += 1

                    if c%100 == 0:
                        print(c,path.join(dir, file))

                    with open(path.join(dir, file)) as f:
                        jsonData = json.load(f)

                    imageName = path.splitext(path.join(path.abspath(dir),file))[0]
                    data = jsonData["annotations"]

                    write = False
                    heads = []

                    if args.merge:
                        trash = False
                        annotations = []
                        for i,datum in enumerate(data):
                            date = datum["date"]
                            date = datetime.strptime(date, DATE_FORMAT)
                            if dateLimit is None or date < dateLimit:
                                ignore = False
                                for name in args.ignore:
                                    if name in datum["annotator"]:
                                        ignore = True
                                if not ignore:
                                    if "trash" in datum:
                                        trash = True
                                        break
                                    annotation = []
                                    if "objects" in data[i]:
                                        for object in datum["objects"]:
                                            if "head" in object:
                                                head = object["head"]
                                                annotation.append([
                                                    head["x1"],
                                                    head["y1"],
                                                    head["x2"],
                                                    head["y2"],
                                                    1
                                                ])
                                    annotations.append(annotation)

                        if not trash:
                            write = True
                            annotation1 = annotations[0]
                            if len(annotations) > 1:
                                # print(len(annotation1),end=" ")
                                for annotation2 in annotations[1:]:
                                    # print(len(annotation2),end=" ")
                                    annotation1 = mergeAnnotations(annotation1,annotation2)
                            # print(np.array(annotation1,dtype=np.uint16)[:,4].sum())

                            heads = np.array(annotation1).flatten().tolist()

                    else:
                        maxDate,maxI = None,None
                        for i,datum in enumerate(data):
                            date = datum["date"]
                            date = datetime.strptime(date, DATE_FORMAT)
                            if (
                                (dateLimit is None or date < dateLimit) and
                                (maxDate is None or maxDate < date)# and datum["annotator"] == "none"
                            ):
                                ignore = False
                                for name in args.ignore:
                                    if name in datum["annotator"]:
                                        ignore = True
                                if not ignore:
                                    maxDate = date
                                    maxI = i

                        if maxI is not None and "trash" not in data[maxI]:
                            write = True
                            annotator = data[maxI]["annotator"].replace(' ','_').replace('\\','_').replace('/','_')
                            if "objects" in data[maxI]:
                                for object in data[maxI]["objects"]:
                                    if "head" in object:
                                        head = object["head"]
                                        heads.extend([
                                            head["x1"],
                                            head["y1"],
                                            head["x2"],
                                            head["y2"],
                                            1
                                        ])

                    # endif

                    if write:
                        imageNames = []
                        if args.count > 1:
                            dirName = path.dirname(imageName)
                            fileName, fileExt = path.splitext(path.basename(imageName))
                            index = fileName.rfind('n')
                            frame = int(fileName[index+1:])
                            fileName = fileName[:index+1]
                            for n in range(-args.count+1,1):
                                editedImageName = path.join( dirName, fileName+str(frame+n)+fileExt )
                                imageNames.append(editedImageName)
                        else:
                            imageNames.append(imageName)

                        csvwriter.writerow(imageNames+heads)
                        if args.annotator:
                            csvwriterAnnotator.writerow(imageNames+[annotator]+heads)

    if args.annotator:
        csvfileAnnotator.close()

def mergeAnnotations(annotation1,annotation2):
    costMatrix = np.ones( (len(annotation1),len(annotation2)), dtype=np.uint8 )

    for i1,box1 in enumerate(annotation1):
        for i2,box2 in enumerate(annotation2):
            if iou(box1,box2) >= IOU_THRESHOLD:
                costMatrix[i1,i2] = 0

    rows, cols = scipy.optimize.linear_sum_assignment(costMatrix)
    selectedIndicesI = rows.tolist()
    selectedIndicesJ = cols.tolist()

    annotation = []
    for i in range(len(annotation1)):
        paired = False
        if i in rows:
            row = i
            col = cols[rows==i][0]

            if costMatrix[row,col] == 0:
                annotation.append([
                    (annotation1[row][0]+annotation2[col][0])/2,
                    (annotation1[row][1]+annotation2[col][1])/2,
                    (annotation1[row][2]+annotation2[col][2])/2,
                    (annotation1[row][3]+annotation2[col][3])/2,
                    annotation1[row][4]])
                paired = True

        if not paired:
            annotation.append(annotation1[i][:-1]+[0])

    for j in range(len(annotation2)):
        if j not in cols:
            annotation.append(annotation2[j][:-1]+[0])

    return annotation

def iou(a,b):
    if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
        return 0

    aW = a[2]-a[0]+1
    aH = a[3]-a[1]+1
    bW = b[2]-b[0]+1
    bH = b[3]-b[1]+1

    aArea = aW*aH
    bArea = bW*bH

    intersection = [
        max(a[0],b[0]),
        max(a[1],b[1]),
        min(a[2],b[2]),
        min(a[3],b[3])
    ]

    intersectionW = max(0,intersection[2]-intersection[0]+1)
    intersectionH = max(0,intersection[3]-intersection[1]+1)
    intersectionArea = intersectionW*intersectionH

    unionArea = aArea+bArea-intersectionArea

    return intersectionArea/unionArea if unionArea !=0 else 0

if __name__ == '__main__':
    main()
