#!/usr/bin/python3

import os.path as path
import os
import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("infiles", nargs="+")
parser.add_argument("-o", "--outdir", default=None)
parser.add_argument("--structure", action="store_true")
parser.add_argument("--threshold", default=0.0, type=float)
args = parser.parse_args()

colors = []
widths = []

d = {}
for csv in args.infiles:
    outFolderName = (
        args.outdir if args.outdir is not None
        else path.splitext(path.basename(csv))[0] + "_annotated_images")

    with open(csv, 'r') as f:
        for line in f:
            line = line.rstrip()
            lineSplit = line.split('\t')
            fileName = lineSplit[0]

            boxes = lineSplit[1:]

            i = 0
            annotation = []
            while i+4 < len(boxes):
                annotation.append([
                    float(boxes[i]),
                    float(boxes[i+1]),
                    float(boxes[i+2]),
                    float(boxes[i+3]),
                    float(boxes[i+4])
                ])
                i += 5

            if fileName in d:
                d[fileName] += [annotation]
            else:
                d[fileName] = [annotation]

for fileName in d:
    srcImagePath = fileName
    if args.structure:
        path.relpath(srcImagePath)
        dstImagePath = path.join(*(
            [outFolderName] + list(path.dirname(srcImagePath).split(os.sep))
            + [path.basename(srcImagePath)]))
    else:
        dstImagePath = path.join(*[
            outFolderName,
            path.dirname(path.relpath(srcImagePath)).replace(os.sep, "__")
            .replace(".", "_").replace("-", "_")+path.basename(srcImagePath)])

    im = cv.imread(srcImagePath, -1)

    for index, annotation in enumerate(d[fileName]):
        for box in annotation:
            x1, y1, x2, y2, c = box
            x1, y1, x2, y2 = (int(e) for e in (x1, y1, x2, y2))
            if c >= args.threshold:
                # color = (0, 0, 255) if c else (255, 0, 0)
                # cv.rectangle(im, (x1, y1), (x2, y2), color, 3)
                color = (255, 0, 0) if index else (0, 0, 255)
                width = 2 if index else 4
                cv.rectangle(im, (x1, y1), (x2, y2), color, width)

    print(srcImagePath, dstImagePath)
    os.makedirs(path.dirname(dstImagePath), exist_ok=True)
    cv.imwrite(dstImagePath, im)
