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

        i = 0
        d[fileName] = []
        while i+4 < len(boxes):
            d[fileName].append([
                int(float(boxes[i])),
                int(float(boxes[i+1])),
                int(float(boxes[i+2])),
                int(float(boxes[i+3])),
                int(float(boxes[i+4]))
            ])
            i += 5

for fileName in d:
    srcImagePath = fileName
    if keepStructure:
        dstImagePath = path.join( *([outFolderName] + list(path.dirname(srcImagePath).split(os.sep)) + [path.basename(srcImagePath)]) )
    else:
        dstImagePath = path.join(*[ outFolderName, path.basename(srcImagePath) ])

    im = cv.imread(srcImagePath, -1)

    for box in d[fileName]:
        x1,y1,x2,y2,c = box
        color = (0,0,255) if c else (255,0,0)
        cv.rectangle(im,(x1,y1),(x2,y2),color,3)

    print(srcImagePath,dstImagePath)
    os.makedirs( path.dirname(dstImagePath), exist_ok=True )
    cv.imwrite( dstImagePath, im )
