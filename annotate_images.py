#!/usr/bin/python3

from sys import argv, exit
import os.path as path
import os, argparse
import cv2 as cv
from pprint import pprint
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("-s", "--keep_structure", action='store_true')
args = parser.parse_args()

csv = args.csv
keepStructure = args.keep_structure
outFolderName = path.splitext(path.basename(csv))[0] + "_annotated_images"

d = {}
with open(csv, 'r') as f:
    for line in f:
        line = line.rstrip()
        lineSplit = line.split('\t')
        fileName = lineSplit[0]

        boxes = lineSplit[1:]
        boxes = [ boxes[i] for i in range(len(boxes)) if not (i+1)%5==0 ]

        i = 0
        d[fileName] = []
        while i+3 < len(boxes):
            d[fileName].append([
                int(boxes[i]),
                int(boxes[i+1]),
                int(boxes[i+2]),
                int(boxes[i+3])])
            i += 4

for fileName in d:
    srcImagePath = fileName
    if keepStructure:
        dstImagePath = path.join( *([outFolderName, path.splitext(path.basename(csv))[0]] + list(path.dirname(srcImagePath).split(os.sep)) + [path.basename(srcImagePath)]) )
    else:
        dstImagePath = path.join(*[ outFolderName, path.splitext(path.basename(csv))[0], path.basename(srcImagePath) ])

    im = cv.imread(srcImagePath, -1)

    for box in d[fileName]:
        x1,y1,x2,y2 = box
        cv.rectangle(im,(x1,y1),(x2,y2),(0,0,255),3)

    print(srcImagePath,dstImagePath)
    os.makedirs( path.dirname(dstImagePath), exist_ok=True )
    cv.imwrite( dstImagePath, im )
