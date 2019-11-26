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
parser.add_argument("-c","--count", default=1, type=int, help="input image count")
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

                    #imageBaseName = jsonData["image"]
                    #imageName = path.join(dir, imageBaseName)
                    imageName = None
                    for result in glob.glob( path.splitext(path.join(dir,file))[0] + ".*" ):
                        if path.splitext(result)[1] != ".json":
                            imageName = result
                    assert imageName is not None

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

                        if "head_det" not in annotator:

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
                        dirName = path.dirname(imageName)
                        fileName, fileExt = path.splitext(path.basename(imageName))
                        index = fileName.rfind('n')
                        frame = int(fileName[index+1:])
                        fileName = fileName[:index+1]
                        for n in range(-args.count+1,1):
                            editedImageName = path.join( dirName, fileName+str(frame+n)+fileExt )
                            imageNames.append(editedImageName)

                        if len(heads) != 0:
                            csvwriter.writerow(imageNames+heads)
                            csvwriterAnnotator.writerow(imageNames+[annotator]+heads)

                        else:
                            csvwriter.writerow(imageNames)
                            csvwriterAnnotator.writerow(imageNames+[annotator])
