import numpy as np
from pprint import pprint
from sys import exit
from PIL import Image
from os import path
import caffe, cv2
caffe.set_random_seed(666)
import numpy.random
numpy.random.seed(666)
import random
random.seed(666)
cv2.setRNGSeed(666)

import util

class RpnDetector:
    def __init__(self, configFile):
        root = path.dirname(configFile)

        self.config = util.readConfig(configFile)

        caffe.set_device(0)
        caffe.set_mode_gpu()

        self.net = caffe.Net( path.join( root, self.config["dnn_deploy_file"][0] ), path.join( root, self.config["dnn_weight_files"][0], ), caffe.TEST )

    def infer(self, imagePath, confidenceThreshold, minHeight, maxHeight):

        # preparing input

        im = cv2.imread(imagePath).astype(self.net.blobs["data"].data.dtype)

        im -= list(map(int,self.config["channel_shift"]))

        pad = int(self.config["pad"][0])
        h,w = im.shape[0:2]

        padH = ( pad - ( h % pad ) ) % pad;
        padW = ( pad - ( w % pad ) ) % pad;

        padded = np.zeros( dtype=self.net.blobs["data"].data.dtype, shape=(h+padH,w+padW,im.shape[2]) )
        padded[:h,:w,:] = im[...]
        im = padded
        h += padH
        w += padW

        im = im.transpose(2,0,1) # nhwcf -> nchw

        im_input = im[np.newaxis, ...]

        self.net.blobs["data"].reshape(*im_input.shape)
        # self.net.blobs["data"].data[...] = im_input

        imSrc = None
        # imSrc = im_input[0,...].copy()
        # imSrc = imSrc.transpose(1,2,0)
        # imSrc += list(map(int,self.config["channel_shift"]))
        #
        # # bgr -> rgb
        # temp = imSrc[...,0].copy()
        # imSrc[...,0] = imSrc[...,2]
        # imSrc[...,2] = temp[...]
        #
        # imSrc[imSrc<0] = 0
        # imSrc = Image.fromarray( imSrc.astype(np.uint8) )

        # running net

        # self.net.forward()
        im_input2 = np.empty(shape=im_input.shape,dtype=im_input.dtype)
        im_input2[...] = im_input[...]
        im_input = im_input2
        forwardKwargs = {"data": im_input.astype(np.float32, copy=False)}
        self.net.forward(**forwardKwargs)

        # processing output

        outScores = self.net.blobs[ self.config["score_blob"][0] ].data
        outBoxes = self.net.blobs[ self.config["bb_reg_blob"][0] ].data

        return outScores, outBoxes, im_input

        scales = self.config["scales"]
        scales = list( map(int, scales) )
        stride = int(self.config["stride"][0])
        lenScales = len(scales)

        boxes, scores = [], []
        for i in range(lenScales):
            for y in range(outScores.shape[2]):
                for x in range(outScores.shape[3]):

                    # anchors & bbox regression

                    currentScore = outScores[0,i+lenScales,y,x]

                    if currentScore > confidenceThreshold:
                        size = scales[i]*stride

                        xCorr = outBoxes[0,4*i,  y,x]*size
                        yCorr = outBoxes[0,4*i+1,y,x]*size
                        wCorr = np.exp(outBoxes[0,4*i+2,y,x])*size
                        hCorr = np.exp(outBoxes[0,4*i+3,y,x])*size

                        xCenter = x*stride+xCorr+stride/2
                        yCenter = y*stride+yCorr+stride/2

                        x1 = xCenter - (wCorr/2)
                        x2 = xCenter + (wCorr/2)
                        y1 = yCenter - (hCorr/2)
                        y2 = yCenter + (hCorr/2)

                        if ( x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and y2 - y1 + 1 >= minHeight and y2 - y1 + 1 <= maxHeight ):
                            boxes.append([x1,y1,x2,y2])
                            scores.append(currentScore)

        if len(scores) == 0:
            return [], [], imSrc

        # grouping detections (nms)

        nmsIouThreshold = float(self.config["iou_threshold"][0])
        groupedScores, groupedBoxes = util.nms(scores, boxes, nmsIouThreshold)

        return groupedScores, groupedBoxes, imSrc
        #return np.array(scores), np.array(boxes), imSrc
