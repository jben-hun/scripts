import numpy as np
import scipy.optimize
import copy
import munkres
import sys

from sys import exit
from pprint import pprint

def nms(scores_, boxes_, threshold):
    scores = np.array(scores_)
    boxes = np.array(boxes_)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iouVal = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iouVal < threshold)[0]
        order = order[inds + 1]

    scores = scores[keep]
    boxes = boxes[keep, :]

    return scores, boxes

def iou(a,b):
    if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
        return 0

    aW = max(0,a[2]-a[0])
    aH = max(0,a[3]-a[1])
    bW = max(0,b[2]-b[0])
    bH = max(0,b[3]-b[1])

    aArea = aW*aH
    bArea = bW*bH

    intersection = [
        max(a[0],b[0]),
        max(a[1],b[1]),
        min(a[2],b[2]),
        min(a[3],b[3])
    ]

    intersectionW = max(0,intersection[2]-intersection[0])
    intersectionH = max(0,intersection[3]-intersection[1])
    intersectionArea = intersectionW*intersectionH

    unionArea = aArea+bArea-intersectionArea

    return intersectionArea/unionArea if unionArea !=0 else 0

def readConfig(configFile):
    config = {}
    with open(configFile, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line == '' or line[0] == '#':
                continue

            line = line.replace(':','')
            lineSplit = line.split(' ')
            if lineSplit[1][0] == '"' and lineSplit[1][-1] == '"':
                lineSplit[1] = lineSplit[1][1:-1]

            if lineSplit[0] in config:
                config[lineSplit[0]].append(lineSplit[1])
            else:
                config[lineSplit[0]] = [lineSplit[1]]

    return config

def readList(listFile):
    testList = []
    with open(listFile, 'r') as f:
        lines = f.readlines()

    lines.sort()
    for line in lines:
        line = line.rstrip()
        if line == '' or line[0] == '#':
            continue

        line = line.replace(' ','')
        lineSplit = line.split('\t')

        boxes_ = [ int(float(lineSplit[i])) for i in range(1,len(lineSplit)) if i%5!=0 ]
        boxes = np.reshape(boxes_,(-1,4)).tolist()

        testList.append({
            "imagePath":lineSplit[0],
            "boxes":boxes
        })

    return testList

def greedyPair(scores, boxes, boxesGt, threshold):
    detections = []

    if len(scores) == 0:
        return detections

    cost = np.empty( shape=(len(boxesGt),len(boxes)), dtype=np.float64 )

    for i,ei in enumerate(boxesGt):
        for j,ej in enumerate(boxes):
            iouVal = iou(ei,ej)
            cost[i][j] = 1 - iouVal

    used = set()
    selectedIndicesI = []
    selectedIndicesJ = []
    for i in range(len(boxesGt)):
        minVal = 0
        minInd = -1
        for j in range(len(boxes)):
            if (j not in used) and (cost[i][j] < minVal or minInd == -1):
                minVal = cost[i][j];
                minInd = j;
        if (minInd != -1):
            used.add(minInd);
            selectedIndicesI.append(i)
            selectedIndicesJ.append(minInd)

    for i in range(len(scores)):
        if i in selectedIndicesI and 1 - cost[i][selectedIndicesJ[selectedIndicesI.index(i)]] > threshold:
            detections.append([
                scores[i],
                1
            ])
        else:
            detections.append([
                scores[i],
                0
            ])

    return detections

def maxPairScipy(scores, boxes, boxesGt, threshold):
    detections = []

    if len(scores) == 0:
        return detections

    cost = np.zeros( shape=(len(boxes),len(boxesGt)), dtype=np.float64 )

    for i,ei in enumerate(boxes):
        for j,ej in enumerate(boxesGt):
            iouVal = iou(ei,ej)
            cost[i][j] = 1 - iouVal

    rows, cols = scipy.optimize.linear_sum_assignment(cost)

    selectedIndicesI = rows.tolist()
    selectedIndicesJ = cols.tolist()

    for i in range(len(scores)):
        if i in selectedIndicesI and 1 - cost[i][selectedIndicesJ[selectedIndicesI.index(i)]] > threshold:
            detections.append([
                scores[i],
                1
            ])
        else:
            detections.append([
                scores[i],
                0
            ])

    return detections

# def maxPairMunkres(scores, boxes, boxesGt, threshold):
#     detections = []
#
#     if len(scores) == 0:
#         return detections
#
#     cost = np.zeros( shape=(len(boxes),len(boxesGt)), dtype=np.float64 ).tolist()
#
#     for i,ei in enumerate(boxes):
#         for j,ej in enumerate(boxesGt):
#             iouVal = iou(ei,ej)
#             cost[i][j] = 1 - iouVal
#
#     m = munkres.Munkres()
#     indices = m.compute(cost)
#
#     selectedIndicesI = []
#     selectedIndicesJ = []
#     for index in indices:
#         selectedIndicesI.append(index[0])
#         selectedIndicesJ.append(index[1])
#
#     for i in range(len(scores)):
#         if i in selectedIndicesI and 1 - cost[i][selectedIndicesJ[selectedIndicesI.index(i)]] > threshold:
#             detections.append([
#                 scores[i],
#                 1
#             ])
#         else:
#             detections.append([
#                 scores[i],
#                 0
#             ])
#
#     return detections

# def maxPair(scores, boxes, boxesGt, threshold):
#     detections = []
#
#     if len(scores) == 0:
#         return detections
#
#     cost = np.zeros( shape=(len(boxes),len(boxesGt)), dtype=np.float64 )
#
#     for i,ei in enumerate(boxes):
#         for j,ej in enumerate(boxesGt):
#             iouVal = iou(ei,ej)
#             if iouVal > threshold:
#                 cost[i,j] = iouVal
#             else:
#                 cost[i,j] = -1
#
#     row_ind, col_ind = assignment(cost)
#
#     for i in range(len(scores)):
#         if -1 != col_ind[i]:
#             detections.append([
#                 scores[i],
#                 1
#             ])
#         else:
#             detections.append([
#                 scores[i],
#                 0
#             ])
#
#     return detections
#
# def assignment(cost):
#     h = cost.shape[0]
#     w = cost.shape[1]
#
#     iSkip = np.where(cost.max(axis = 1) == -1)
#     jSkip = np.where(cost.max(axis = 0) == -1)
#
#     state = []
#     exit = False
#     depth = 1
#
#     row_ind = list(range(h))
#
#     maxVal = None
#
#     def change(index,state):
#         if index >= len(state):
#             state += [-1]
#             return True
#
#         if np.isin(index, iSkip):
#             return False
#
#         val = state[index]
#         val += 1
#         while val < w:
#             if cost[index,val] != -1 and not np.isin(val, jSkip) and val not in state:
#                 state[index] = val
#                 return True
#             val += 1
#
#         return False
#     # end of function definition
#
#     while not exit:
#         success = change(index=depth-1, state=state)
#
#         if success:
#             if depth == h:
#                 rowIx = np.arange(h)
#                 colIx = np.array(state)
#                 keep = np.where(colIx != -1)
#                 rowIx = rowIx[keep]
#                 colIx = colIx[keep]
#                 newMaxVal = cost[rowIx,colIx].sum()
#                 if maxVal is None:
#                     col_ind = copy.deepcopy(state)
#                     maxVal = newMaxVal
#                 elif newMaxVal > maxVal:
#                     col_ind = copy.deepcopy(state)
#                     maxVal = newMaxVal
#
#             else:
#                 depth += 1
#         else:
#             state = state[0:depth-1]
#             depth -= 1
#             if depth == 0:
#                 exit = True
#
#     return row_ind, col_ind

# def assignmentOld(cost):
#     h = cost.shape[0]
#     w = cost.shape[1]
#
#     iSkip = np.where(cost.max(axis = 1) == -1)
#     jSkip = np.where(cost.max(axis = 0) == -1)
#     # print(cost.shape)
#     # print(cost.shape[0]-len(iSkip), cost.shape[1]-len(jSkip))
#
#     cost.tofile('foo.csv',sep=',',format='%10.5f')
#
#     skipped = 0
#     result = [[]]
#     for i in range(cost.shape[0]):
#         if np.isin(i, iSkip):
#             skipped += 1 # todo
#             print(skipped)
#             newResult = []
#             for k,v in enumerate(result):
#                 newResult.append(v+[-1])
#
#             result = copy.deepcopy(newResult)
#
#         else:
#             newResult = []
#             for j in range(cost.shape[1]):
#                 for k,v in enumerate(result):
#                     newResult.append(v+[-1])
#                     if not np.isin(j, jSkip):
#                         if j not in v and cost[i,j] != -1:
#                             newResult.append(v+[j])
#                     else:
#                         skipped += 1 # todo
#                         print(skipped)
#
#             result = copy.deepcopy(newResult)
#
#     chosenResult = result[-1]
#     # row_ind = []
#     # col_ind = []
#     # for i in range(h):
#     #     if chosenResult[i] != -1:
#     #         row_ind.append(i)
#     #         col_ind.append(chosenResult[i])
#     row_ind = list(range(h))
#     col_ind = chosenResult
#
#     return row_ind, col_ind
