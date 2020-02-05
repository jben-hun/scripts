#! /usr/bin/python3

import json, os, csv, argparse
import os.path as path
from pprint import pprint
from datetime import datetime
import sys, glob
from math import ceil, floor
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("-c","--count",default=1,type=int,help="input image count")
parser.add_argument("-i","--ignore",nargs='+',default=["head_det"], help="annotator(s) to ignore")
args = parser.parse_args()

oldDate = datetime.strptime("1900", "%Y")

name = args.path.replace(os.sep,"")

with open( name + ".csv", 'w', newline='' ) as csvfile:
    csvwriter = csv.writer(csvfile, delimiter="\t", lineterminator="\n")

    with open( name + "_annotator.csv", 'w', newline='' ) as csvfileAnnotator:
        csvwriterAnnotator = csv.writer(csvfileAnnotator, delimiter="\t", lineterminator="\n")

        for (dir, dirs, filenames) in os.walk(args.path):
            for file in filenames:
                if ".json" in file and ".log.json" not in file:
                    jsonString = ""

                    with open(path.join(dir, file)) as f:
                        jsonData = json.load(f)

                    # 1
                    # imageBaseName = jsonData["image"]
                    # imageName = path.join(dir, imageBaseName)

                    # 2
                    # imageName = None
                    # for result in glob.glob( path.splitext(path.join(dir,file))[0] + ".*" ):
                    #     if path.splitext(result)[1] != ".json":
                    #         imageName = result
                    # assert imageName is not None

                    # 3
                    imageName = path.splitext(path.join(dir,file))[0]

                    data = jsonData["annotations"]

                    maxDate = oldDate
                    maxI = -1
                    for i in range(len(data)):
                        date = data[i]["date"]
                        date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")
                        if date >= maxDate:
                            maxDate = date
                            maxI = i

                    if maxI != -1:
                        heads = []
                        annotator = data[maxI]["annotator"].replace(' ','_').replace('\\','_').replace('/','_')

                        ignore = False
                        for name in args.ignore:
                            if name in annotator:
                                ignore = True

                        if not ignore:
                            for i in range(len(data[maxI]["objects"])):
                                object = data[maxI]["objects"][i]
                                if "head" in object:
                                    head = object["head"]

                                    heads.extend([
                                        head["x1"],
                                        head["y1"],
                                        head["x2"],
                                        head["y2"],
                                        1
                                    ])

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
                        csvwriterAnnotator.writerow(imageNames+[annotator]+heads)
