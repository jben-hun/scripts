#!/usr/bin/python3

from sys import argv
import os.path as path
import os
import cv2 as cv

truthCsv = argv[1]
goodCsv = argv[2]
# badCsv = argv[3]

d = {}

with open(goodCsv, 'r') as f:
    for line in f:
        line = line.rstrip()
        lineSplit = line.split('\t')
        fileName = lineSplit[0]

        boxes = lineSplit[1:]
        boxes = [boxes[i] for i in range(len(boxes)) if not (i+1) % 5 == 0]

        i = 0
        _boxes = []
        while i+3 < len(boxes):
            _boxes.append([
                float(boxes[i]),
                float(boxes[i+1]),
                float(boxes[i+2]),
                float(boxes[i+3])
            ])

            i += 4

        boxes = _boxes

        if fileName not in d:
            d[fileName] = {}

        d[fileName]["good"] = boxes[:]

# with open(badCsv, 'r') as f:
#     for line in f:
#         line = line.rstrip()
#         lineSplit = line.split('\t')
#         fileName = lineSplit[0]
#
#         boxes = lineSplit[1:]
#         boxes = [boxes[i] for i in range(len(boxes)) if not (i+1) % 5 == 0]
#
#         i = 0
#         _boxes = []
#         while i+3 < len(boxes):
#             _boxes.append([
#                 float(boxes[i]),
#                 float(boxes[i+1]),
#                 float(boxes[i+2]),
#                 float(boxes[i+3])
#             ])
#
#             i += 4
#
#         boxes = _boxes
#
#         if fileName not in d:
#             d[fileName] = {}
#
#         d[fileName]["bad"] = boxes[:]

with open(truthCsv, 'r') as f:
    for line in f:
        line = line.rstrip()
        lineSplit = line.split('\t')
        fileName = lineSplit[0]

        boxes = lineSplit[1:]
        boxes = [boxes[i] for i in range(len(boxes)) if not (i+1) % 5 == 0]

        i = 0
        _boxes = []
        while i+3 < len(boxes):
            _boxes.append([
                float(boxes[i]),
                float(boxes[i+1]),
                float(boxes[i+2]),
                float(boxes[i+3])
            ])

            i += 4

        boxes = _boxes

        if fileName not in d:
            d[fileName] = {}

        d[fileName]["truth"] = boxes[:]

badCount = 0
goodCount = 0
truthCount = 0
fileCount = 0
for fileName in d:
    fileCount += 1
    srcImagePath = fileName
    dstImagePath = path.join(*(
        ["annotated_images"] + list(path.dirname(srcImagePath).split(os.sep))
        + [path.basename(srcImagePath)]))
    dstImagePath = path.join("annotated_images", path.basename(srcImagePath))

    im = cv.imread(srcImagePath, -1)

    if "truth" in d[fileName]:
        for box in d[fileName]["truth"]:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            x1, y1, x2, y2 = (
                int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            cv.rectangle(im, (x1, y1), (x2, y2), (128, 128, 128), 5)
            truthCount += 1

    if "good" in d[fileName]:
        for box in d[fileName]["good"]:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            x1, y1, x2, y2 = (
                int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            cv.rectangle(im, (x1, y1), (x2, y2), (0, 128, 0), 2)
            goodCount += 1

    # if "bad" in d[fileName]:
    #     for box in d[fileName]["bad"]:
    #         x1 = box[0]
    #         y1 = box[1]
    #         x2 = box[2]
    #         y2 = box[3]
    #         x1, y1, x2, y2 = (
    #             int(round(x1)), int(round(y1)), int(round(x2)),
    #             int(round(y2)))
    #         cv.rectangle(im, (x1, y1), (x2, y2), (0, 0, 128), 2)
    #         badCount += 1

    print(srcImagePath, dstImagePath)
    os.makedirs(path.dirname(dstImagePath), exist_ok=True)
    cv.imwrite(dstImagePath, im)

# print(
#     "bad:",
#     badCount,
#     "good:",
#     goodCount,
#     "TAR:",
#     goodCount/truthCount,
#     "truth:",
#     truthCount
# )
