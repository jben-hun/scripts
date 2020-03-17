import numpy as np
import scipy.optimize
from sys import exit
from pprint import pprint

def iou(a,aType,b,bType):
    def transformBody(body,w,h):
        d = max(0,h-w)
        body[3] -= d
        h -= d

    if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
        return 0

    aBody,bBody=False,False
    if aType != bType:
        if aType == "body":
            aBody = True
        else:
            bBody = True

    aW = a[2]-a[0]+1
    aH = a[3]-a[1]+1
    bW = b[2]-b[0]+1
    bH = b[3]-b[1]+1

    if aBody:
        transformBody(a,aW,aH)
    elif bBody:
        transformBody(b,bW,bH)

    intersection = [
        max(a[0],b[0]),
        max(a[1],b[1]),
        min(a[2],b[2]),
        min(a[3],b[3])
    ]

    intersectionW = max(0,intersection[2]-intersection[0]+1)
    intersectionH = max(0,intersection[3]-intersection[1]+1)
    intersectionArea = intersectionW*intersectionH

    if aBody:
        bArea = bW*bH
        unionArea = bArea
    elif bBody:
        aArea = aW*aH
        unionArea = aArea
    else:
        aArea = aW*aH
        bArea = bW*bH
        unionArea = aArea+bArea-intersectionArea

    return intersectionArea/unionArea if unionArea !=0 else 0

def readList(listFile,count,detection=False):
    testList = []
    with open(listFile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        if line == '' or line[0] == '#':
            continue

        line = line.replace(' ','')
        lineSplit = line.split('\t')

        if detection:
            boxes_ = [ float(lineSplit[i]) for i in range(1,len(lineSplit)) ]
            boxes_ = [ int(v) if (i+1)%5!=0 else v for i,v in enumerate(boxes_) ]
            boxes = np.reshape( boxes_, (-1,5) )
            scores = boxes[:,4].tolist()
            boxes = boxes[:,:4].tolist()
            testList.append({
                "image_path":lineSplit[0],
                "boxes":boxes,
                "scores":scores
            })
        else:
            boxes_ = [ float(lineSplit[i]) for i in range(count,len(lineSplit)) ]
            boxes_ = [ int(v) if (i+1)%5!=0 else v for i,v in enumerate(boxes_) ]
            boxes = np.reshape( boxes_, (-1,5) )
            labels = boxes[:,4].tolist()
            boxes = boxes[:,:4].tolist()
            testList.append({
                "image_path":lineSplit[count-1],
                "boxes":boxes,
                "labels":labels
            })

    testList.sort(key=lambda x: x["image_path"])

    return testList


def iou_cost(detection_box, detection_type, annotation_box, annotation_type):
    iouVal = iou(detection_box, detection_type, annotation_box, annotation_type)
    return -iouVal


def get_height(target, target_type):
    if target_type in ("head", "body"):
        return target[3]-target[1] + 1
    else:
        raise NotImplemented


def is_height_in_range(target, target_type, minHeight, maxHeight):
    height = get_height(target, target_type)
    lower_inside = minHeight is None or minHeight <= height
    upper_inside = maxHeight is None or maxHeight >= height
    return lower_inside and upper_inside


def max_pair(detections, detection_scores, detection_type, annotations, annotation_labels, annotation_type,
             score_threshold, minHeight=None, maxHeight=None, mask=False, cost_fun=iou_cost):
    classified_detections = []

    num_of_detections = len(detections)
    num_of_annotations = len(annotations)

    if not num_of_detections:
        return classified_detections

    cost = np.zeros(shape=(num_of_detections, num_of_annotations), dtype=np.float64)

    for det_ind, det in enumerate(detections):
        for ann_ind, ann in enumerate(annotations):
            cost[det_ind][ann_ind] = cost_fun(det, detection_type, ann, annotation_type)

    rows, cols = scipy.optimize.linear_sum_assignment(cost)

    selectedIndicesI = rows.tolist()
    selectedIndicesJ = cols.tolist()

    for i in range(len(detection_scores)):
        if i in selectedIndicesI:
            j = selectedIndicesJ[selectedIndicesI.index(i)]
            if mask and annotation_labels[j] == 0:
                continue
            if cost[i][j] < score_threshold:
                if not is_height_in_range(annotations[j], annotation_type, minHeight, maxHeight):
                    continue
                truePositive = True
            else:
                truePositive = False
        else:
            if not is_height_in_range(detections[i], annotation_type, minHeight, maxHeight):
                continue
            truePositive = False

        classified_detections.append([detection_scores[i], int(truePositive)])

    return classified_detections


def maxPair(scores, boxes, detectionType, labels, boxesGt, annotationType, threshold, minHeight=None, maxHeight=None,
            mask=False):
    detections = []

    if len(scores) == 0:
        return detections

    cost = np.zeros( shape=(len(boxes),len(boxesGt)), dtype=np.float64 )

    for i,ei in enumerate(boxes):
        for j,ej in enumerate(boxesGt):
            iouVal = iou(ei,detectionType,ej,annotationType)
            cost[i][j] = 1 - iouVal

    rows, cols = scipy.optimize.linear_sum_assignment(cost)

    selectedIndicesI = rows.tolist()
    selectedIndicesJ = cols.tolist()

    for i in range(len(scores)):
        if i in selectedIndicesI:
            j = selectedIndicesI.index(i)
            if mask and labels[selectedIndicesJ[j]] == 0:
                continue
            if 1 - cost[i][selectedIndicesJ[j]] > threshold:
                if minHeight is not None or maxHeight is not None:
                    selectedIndex = selectedIndicesJ[j]
                    selectedGt = boxesGt[selectedIndex]
                    selectedHeight = selectedGt[3] - selectedGt[1] + 1
                    if (minHeight is not None and selectedHeight < minHeight) or \
                        (maxHeight is not None and selectedHeight > maxHeight):
                        continue
                truePositive = True
            else:
                truePositive = False

        else:
            if minHeight is not None or maxHeight is not None:
                selectedIndex = i
                selected = boxes[selectedIndex]
                selectedHeight = selected[3] - selected[1] + 1
                if (minHeight is not None and selectedHeight < minHeight) or \
                    (maxHeight is not None and selectedHeight > maxHeight):
                    continue
            truePositive = False

        if truePositive:
            detections.append([scores[i],1])
        else:
            detections.append([scores[i],0])

    return detections

def heightFilter(l_,minHeight,maxHeight,labels=None):
    l = l_[:]

    if labels is not None:
        l = [ e for i,e in enumerate(l) if labels[i] != 0 ]

    if minHeight is not None:
        l = [ e for e in l if e[3]-e[1]+1 >= minHeight ]
    if maxHeight is not None:
        l = [ e for e in l if e[3]-e[1]+1 <= maxHeight ]

    return l
