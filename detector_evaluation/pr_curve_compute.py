#!/usr/bin/python3

import os
import csv
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotations", type=str, required=True)
parser.add_argument("-d", "--detections", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-m", "--bb_match_metric", type=str, choices=["IOU", "IOA", "IOD", "SPIA"], default="IOU")
parser.add_argument("-t", "--bb_match_threshold", type=float, default=0.5)
parser.add_argument("--square_annotations", action="store_true")
parser.add_argument("--square_detections", action="store_true")
parser.add_argument("--delimiter", type=str, default="\t")
args = parser.parse_args()


annotationsPath = args.annotations
detectionsPath = args.detections
outputPath = args.output
matchMetric = args.bb_match_metric
matchTreshold = args.bb_match_threshold
squareAnns = args.square_annotations
squareDets = args.square_detections
delimiter = args.delimiter

confIdxIncrement = 1

def readCSVFile(csvPath):
  annotationsDict = {}

  with open(csvPath) as csvFile:
    reader = csv.reader(csvFile, delimiter = delimiter)

    for row in reader:
      start = 1
      bbs = annotationsDict.setdefault(row[0], [])

      if len(row) <= 2:   # Skipping "imageName" and "imageName\t" lines
        continue

      while start < len(row):
        bbs.append(list(map(float, row[start : start + 5])))
        start += 5

  return annotationsDict

def readSkeletonJSONFile(jsonPath):
  with open(jsonPath, "rb") as fp:
    skeletonDataset = json.load(fp)

  output = {}
  for skeletonFile in skeletonDataset:
    assert skeletonFile["filename"] not in output
    output[skeletonFile["filename"]] = []
    for skel in skeletonFile["skeletons"]:
      if not len(skel):
        continue
      x = [pt["x"] for pt in skel]
      y = [pt["y"] for pt in skel]
      scores = [pt["confidence"] for pt in skel]
      bb = [min(x), min(y), max(x), max(y), sum(scores)/len(scores), list(zip(x, y))]
      output[skeletonFile["filename"]].append(bb)
  return output


def getIOU(annBB, predBB, matchMetric):
  minXpred, minYpred, maxXpred, maxYpred = predBB[0:4]
  minXann, minYann, maxXann, maxYann = annBB[0:4]

  if maxXpred < minXann:
    return 0

  if maxYpred < minYann:
    return 0

  if maxXann < minXpred:
    return 0

  if maxYann < minYpred:
    return 0

  xI = max(minXpred, minXann)
  yI = max(minYpred, minYann)

  wI = min(maxXpred, maxXann) - xI
  hI = min(maxYpred, maxYann) - yI

  wPred = maxXpred - minXpred
  hPred = maxYpred - minYpred

  wAnn = maxXann - minXann
  hAnn = maxYann - minYann

  areaI = wI * hI
  areaPred = wPred * hPred
  areaAnn = wAnn * hAnn

  if matchMetric == 'IOD':
    return areaI / areaPred

  if matchMetric == 'IOA':
    return areaI / areaAnn

  if matchMetric == 'IOU':
    return areaI / (areaPred + areaAnn - areaI)

  if matchMetric == "SPIA":
    inside = [
      (x >= minXann) and (x <= maxXann) and
      (y >= minYann) and (y <= maxYann) for x, y in predBB[5]
    ]
    return sum(inside)/len(inside) if len(inside) else 0


def getImgMetrics(predItems, annBBs, ignoreAnnBBs, scoreTresh, mode):
  ious = []

  posPreds = 0

  for predBox, pIdx in zip(predItems, range(len(predItems))):
    if predBox[4] >= scoreTresh:
      posPreds += 1
      for annBox, aIdx in zip(annBBs, range(len(annBBs))):
        iou = getIOU(annBox, predBox, mode)

        if iou >= matchTreshold:
          ious.append((iou, pIdx, aIdx))

  ious.sort(key = lambda x: -x[0])

  p_used = set()
  a_used = set()

  for _, pIdx, aIdx in ious:
    if pIdx in p_used or aIdx in a_used:
      continue

    p_used.add(pIdx)
    a_used.add(aIdx)

  if len(ignoreAnnBBs) != 0:
    for predBox, pIdx in zip(predItems, range(len(predItems))):
      if predBox[4] < scoreTresh:
        continue

      if pIdx in p_used:
        continue

      for annBox in ignoreAnnBBs:
        iou = getIOU(annBox, predBox, mode)

        if iou >= matchTreshold:
          posPreds -= 1
          break


  return len(a_used), posPreds - len(p_used), len(annBBs) - len(a_used)

def getGraphValues(activeAnnsDict, ignoreAnnsDict, predsDict, matchMetric):
  tps = []
  fps = []
  fns = []
  scores = []
  recalls = []
  precisions = []

  predScores = []

  imgStatsDict = {}

  cTP = 0
  cFP = 0
  cFN = 0

  for img, boxes in predsDict.items():
    annsL = len(activeAnnsDict[img])
    cFN += annsL
    imgStatsDict[img] = (0, 0, annsL)
    for box in boxes:
      predScores.append((box[4], img))

  predScores.sort(key = lambda x: -x[0])

  for i in range(len(predScores)):
    scoreTresh, imgId = predScores[i]
    oldImgStats = imgStatsDict[imgId]
    newImgStats = getImgMetrics(predsDict[imgId], activeAnnsDict[imgId], ignoreAnnsDict[imgId], scoreTresh, matchMetric)

    cTP += newImgStats[0] - oldImgStats[0]
    cFP += newImgStats[1] - oldImgStats[1]
    cFN += newImgStats[2] - oldImgStats[2]

    imgStatsDict[imgId] = newImgStats

    if i % confIdxIncrement == 0 or i == (len(predScores) - 1):
      tps.append(cTP)
      fps.append(cFP)
      fns.append(cFN)
      scores.append(scoreTresh)

      recalls.append( cTP / (cTP + cFN) if (cTP + cFN)>0 else 0.0 )
      precisions.append( cTP / (cTP + cFP) if (cTP + cFP)>0 else 1.0 )

  return scores, precisions, recalls

def squareBoxes(bbDict):
  for imageBBs in bbDict.values():
    for bb in imageBBs:
      if 1 == bb[4]:
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]

        if h > w:
          bb[3] = bb[1] + w


def seperateAnnsDict(annsDict):
  activeAnnsDict = {}
  ignoreAnnsDict = {}

  for img, boxes in annsDict.items():
    activeBBs = activeAnnsDict.setdefault(img, [])
    ignoreBBs = ignoreAnnsDict.setdefault(img, [])

    for box in boxes:
      if box[4] != 1:
        ignoreBBs.append(box)
      else:
        activeBBs.append(box)

  return activeAnnsDict, ignoreAnnsDict

annotationsDict = readCSVFile(annotationsPath)
detectionsExt = os.path.splitext(detectionsPath)[1].lower()
if detectionsExt == ".csv":
  predictionsDict = readCSVFile(detectionsPath)
elif detectionsExt == ".json":
  predictionsDict = readSkeletonJSONFile(detectionsPath)
else:
  raise ValueError("Unknown detection file type!")

if squareAnns:
  squareBoxes(annotationsDict)

if squareDets:
  squareBoxes(predictionsDict)

activeAnnsDict, ignoreAnnsDict = seperateAnnsDict(annotationsDict)

scores, precisions, recalls = getGraphValues(activeAnnsDict, ignoreAnnsDict, predictionsDict, matchMetric)

with open(outputPath, "w") as outputFile:
  writer = csv.writer(outputFile, delimiter = "\t")

  for row in zip(scores, precisions, recalls):
    writer.writerow(row)
