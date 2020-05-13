import copy
import argparse
import os
import numpy as np
from munkres import Munkres
import collections
from math import exp

import caffe
import xml.etree.ElementTree
import cv2




np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

def IOU_rects(r1, r2):
    SI = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0])) * max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    S1 = (r1[2]-r1[0])*(r1[3]-r1[1])
    S2 = (r2[2]-r2[0])*(r2[3]-r2[1])
    S = S1+S2-SI
    return SI/S


def nms(dets, thresh):
    if not dets.shape[0]:
      return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep





def calcWiseMatrix(annots, dets):
    wise = []
    for annot in annots:
        wise_row = []
        for det in dets:
            wise_row.append(1-IOU_rects(annot, det))
        wise.append(wise_row)
    return wise


def vis_detections_opencv(im, class_name, dets, color ,thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), font, 1, color, 1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='path_net', help='Path of Proto')
    parser.add_argument('--model', dest='path_model', help='Path of .caffemodel')
    parser.add_argument('--imdb_path',dest='path_imdb')
    args = parser.parse_args()

    return args


def read_gt_boxes(annotate_file):
    bboxes = []
    e = xml.etree.ElementTree.parse(annotate_file).getroot()
    for obj in e.findall('object'):
        bbox= obj.find('bndbox')
        if bbox == None:
            continue
        numpy_bbox= [ float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text), float(bbox.find('ymax').text), 1.0]
        bboxes.append( numpy_bbox)
    return bboxes



def im_dete(net, frame , threshold):
    im_orig = frame.astype(np.float32, copy=True)
    im_orig -= [104,117,123]

    pad_h = 32 - im_orig.shape[0] % 32
    pad_h =  0 if pad_h is 32 else pad_h

    pad_w = 32 - im_orig.shape[1] % 32
    pad_w =  0 if pad_w is 32 else pad_w

    blob = np.zeros((1, im_orig.shape[0]+pad_h, im_orig.shape[1]+pad_w, 3),
                    dtype=np.float32)
    blob[0, 0:im_orig.shape[0], 0:im_orig.shape[1], :] = im_orig
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    net.blobs['data'].reshape(*(blob.shape))
    forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
    net.forward(**forward_kwargs)



    blob_scores = net.blobs['rpn_16_cls_prob_reshape'].data.copy()
    blob_boxes = net.blobs['rpn_16_bbox_pred'].data.copy()
    boxes = []
    scores= []
    scales = [2,4,8,16]
    for i in range(0, len(scales)):
        for y in range(0,blob_scores.shape[2]):
            for x in range(0, blob_scores.shape[3]):
                if blob_scores[0,4+i,y,x] > threshold:
                    scores.append(blob_scores[0,4+i,y,x])
                    size = scales[i]*16
                    x_corr = blob_boxes[0,4*i,y,x] * size
                    y_corr = blob_boxes[0,4*i+1,y,x] * size
                    w_pred = exp(blob_boxes[0,4*i+2,y,x]) * size
                    h_pred = exp(blob_boxes[0,4*i+3,y,x]) * size

                    center_x = x*16+8 + x_corr
                    center_y = y*16+8 + y_corr
                    boxes.append([center_x-w_pred/2,center_y-h_pred/2,center_x+w_pred/2,center_y+h_pred/2])


    blob_scores = net.blobs['rpn_8_cls_prob_reshape'].data.copy()
    blob_boxes = net.blobs['rpn_8_bbox_pred'].data.copy()
    scales = [2,4]
    for i in range(0, len(scales)):
        for y in range(0,blob_scores.shape[2]):
            for x in range(0, blob_scores.shape[3]):
                if blob_scores[0,2+i,y,x] > threshold:
                    scores.append(blob_scores[0,2+i,y,x])
                    size = scales[i]*8
                    x_corr = blob_boxes[0,4*i,y,x] * size
                    y_corr = blob_boxes[0,4*i+1,y,x] * size
                    w_pred = exp(blob_boxes[0,4*i+2,y,x]) * size
                    h_pred = exp(blob_boxes[0,4*i+3,y,x]) * size

                    center_x = x*8+4 + x_corr
                    center_y = y*8+4 + y_corr
                    boxes.append([center_x-w_pred/2,center_y-h_pred/2,center_x+w_pred/2,center_y+h_pred/2])

    return scores, boxes

if __name__ == '__main__':
    CONF_THRESH = 0.2
    NMS_THRESH = 0.3


    args = parse_args()


    if not os.path.isfile(args.path_model):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(args.path_model))


    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.path_net, args.path_model, caffe.TEST)

    imdb = os.path.join(args.path_imdb,'Splits/test.txt')

    full_annot_num = 0
    full_det_list=[]

    matcher = Munkres()
    with open(imdb) as f:
        pic_num=0
        for picture_name in f:
            pic_num+=1
            print( pic_num)
            frame_path=os.path.join(args.path_imdb,'JPEGImages',picture_name.rstrip('\n')+".jpeg")
            gt_boxes = read_gt_boxes(os.path.join(args.path_imdb,'Annotations',picture_name.rstrip('\n')+".xml"))
            gt_boxes = np.array(gt_boxes)
            if len(gt_boxes)==0:
                continue


            frame = cv2.imread(frame_path)

            scores, boxes = im_dete(net, frame, CONF_THRESH)

            #get bboxes of second class

            cls_boxes  = np.array(boxes)
            cls_scores = np.array(scores)
            print (cls_boxes.shape, cls_scores.shape)

            dets = np.empty([0,5])
            if cls_boxes.shape[0]:
                dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            delete_num=0
            for ind in np.where(dets[:, -1] < CONF_THRESH)[0]:
                dets=np.delete(dets,(ind-delete_num),axis=0)
                delete_num+=1

            full_annot_num+=len(gt_boxes)

            if len(dets)==0 :
                continue
            wise = calcWiseMatrix(gt_boxes, dets)
            #print "dets:", len(dets)
            #print "annots:", len(gt_boxes)
            #print wise

            indexes = matcher.compute(wise)
            #print indexes
            for i in range(0, len(dets)):
                b = [item for item in indexes if item[1] == i]
                if len(b) == 1 and wise[b[0][0]][b[0][1]] <= 0.5:
                    full_det_list.append((dets[i][4],True))
                else:
                    full_det_list.append((dets[i][4],False))
        full_det_list.sort(key=lambda tup: tup[0])
        output = open("report_faster.csv", 'w')
        bad_num = 0
        found_num = 0
        for x in reversed(full_det_list):
            if x[1]:
                found_num+=1
            else:
                bad_num+=1
            output.write('%f\t%d\t%f\n' % (float(found_num)/float(full_annot_num), bad_num, x[0]))
