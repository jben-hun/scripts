#! /usr/bin/python3

import sys, os
from PIL import Image, ImageDraw
import numpy as np
from sys import exit
import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 28})

# fig, ax = plt.subplots()

countSet = {}

path = sys.argv[1]

with open(path, "r") as f:
    lines = f.readlines()

d = {}
dCount = {}
for line in lines:
    if len(line.split('\t')) == 1:
        continue
    key = '/'.join((line.split('\t')[0]).split('/')[:-1])
    key = key.replace('/','_')
    if "rossmann" in key:
        key = "rossmann"
    elif "film" in key:
        key = "film"
    elif "10.100.22." in key:
        key = "budapestpark"
    elif "Elephant" in key:
        key = "elephant"
    elif "HollywoodHeads" in key:
        key = "hollywoodheads"
    elif "MPII" in key:
        key = "mpii"
    elif "(2665)" in key:
        key = "elms"
    elif "brainwash" in key:
        key = "brainwash"

    boxes = line[:-1].split('\t')[1:]
    boxes = [i for i in boxes if i != '']
    boxes = list(map(float, boxes))
    boxNum = len(boxes)//5
    for i in range(boxNum):
        index = i*5
        x1 = boxes[index]
        y1 = boxes[index+1]
        x2 = boxes[index+2]
        y2 = boxes[index+3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        boxSize = h

        if min(h,w) < 10:
            # path = "/home/bjenei/mount/research01/"+line.split('\t')[0]
            # filename = path.split('/')[-1]
            # im = Image.open(path)
            # draw = ImageDraw.Draw(im)
            # draw.rectangle([(x1,y1),(x2,y2)],outline=(255,0,0,255))
            # del draw
            # os.makedirs(key, exist_ok=True)
            # im.save(key + '/' + filename)

            continue

        if key in d:
            d[key].append(boxSize)
        else:
            d[key] = [boxSize]

    if key in d:
        if key in dCount:
            dCount[key] += 1
        else:
            dCount[key] = 1

sortedKeys = sorted(d.keys(),key=str.lower)

for key in sortedKeys:
    # sizes = sorted(d[key])
    # print(key)
    # sizes = list(filter(lambda x: x > 0, sizes))
    # print(len(sizes))
    # print(sizes[:20])
    # print()

    a = np.array(d[key])
    countSize = a.shape[0]
    meanBoxNum = countSize / dCount[key]
    meanSize = np.mean(a,dtype=np.float64)
    stdSize = np.std(a,dtype=np.float64)
    line = "images: {:7.0f}   boxes: {:8.0f}   bb/img: {:3.1f}   size: {:4.0f}   std: {:4.0f}   @   {}".format(
        dCount[key],
        countSize,
        meanBoxNum,
        meanSize,
        stdSize,
        key
    )

    print(line)

    # ax.hist(
    #     a,
    #     color='g',
    #     bins=int(round( (np.max(a)-np.min(a)+1)/10 ))
    # )
    # plt.title(key)
    # ax.set_xlim(left=-50,right=600)
    # ax.set_xticks(list(range(-50,650,50)))
    # fig.set_size_inches(16, 8) # w,h
    # fig.savefig("histogram_"+key+".png")
    # plt.cla()


"""
v3:
/home/jlocki/data/HollywoodHeads/JPEGImages    43344
/home/jlocki/data/MPII/images                  10724 nem kell (?)
/home/jlocki/data/UltiCorpus/film1             421
/home/jlocki/data/UltiCorpus/film2             541
/home/jlocki/data/UltiCorpus/film3             685
/home/jlocki/data/UltiCorpus/film4             575
/home/jlocki/data/UltiCorpus/film5             542
/home/jlocki/data/UltiCorpus/film6             611
/home/jlocki/data/UltiCorpus/film7             474
/home/jlocki/data/UltiCorpus/film8             886
/home/jlocki/data/UltiCorpus/film9             475
/home/jlocki/data/UltiCorpus/rossmann_cash1    3364
/home/jlocki/data/UltiCorpus/rossmann_cash2    4366
/home/jlocki/data/UltiCorpus/rossmann_entrance 2144
/home/jlocki/data/UltiCorpus/rossmann_line     2947
total: 72099

v3+bjenei:
coco:      + 22927
sequences: + 6619
total: 90921

v2:
/home/bogyom/ultinous/data/COCO/train2014 29105 - 9616 = 19489
/home/bogyom/ultinous/data/HollywoodHeads/JPEGImages 216694
245799
"""
