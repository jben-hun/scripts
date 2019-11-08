#!/usr/bin/python3

import numpy as np
from sys import exit
from math import floor, ceil
import os.path as path
from pprint import pprint
import os, argparse

report = {}

def error(str):
    if str in report:
        report[str] += 1
    else:
        report[str] = 1
    print(str)
    print(line)
    print()


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("-c","--count", default=1, type=int, help="input image count")
args = parser.parse_args()

file = args.path
count = args.count

with open(file,'r') as f:
    lines = f.readlines()

lines = [ line[:-1] for line in lines ]

with open( path.splitext(file)[0]+"_filtered"+path.splitext(file)[1], 'w' ) as f:
    for line in lines:
        splitLine = line.split('\t')
        image = splitLine[count-1]
        images = splitLine[:count]

        if not path.isfile(image) or path.getsize(image) <= 0:
            error("file: skipped")
            continue

        boxes = splitLine[count:]
        if len(boxes) %5!= 0:
            error("boxdiv: skipped")
            continue

        boxes = [ float(e) for e in boxes ]
        boxes = np.reshape(boxes,(-1,5)).tolist()

        i = 0
        while i < len(boxes):
            box = boxes[i]
            boxes[i][0] = int(floor(box[0]))
            boxes[i][1] = int(floor(box[1]))
            boxes[i][2] = int(ceil(box[2]))
            boxes[i][3] = int(ceil(box[3]))
            boxes[i][4] = 1

            if boxes[i][0] >= boxes[i][2] or boxes[i][1] >= boxes[i][3]:
                error("coordflip: skipped")
                del boxes[i]
                continue

            if boxes[i][4] != 1:
                error("coordone: skipped")
                del boxes[i]
                continue

            if boxes[i] in boxes[:i]:
                error("duplicate: skipped")
                del boxes[i]
                continue

            i += 1

        if len(boxes) == 0:
            error("boxnum: skipped")
            continue

        if count > 1:
            text = '\t'.join(images)
        else:
            text = image

        for box in boxes:
            for e in box:
                text += '\t' + str(e)

        f.write(text+'\n')

    pprint(report)
