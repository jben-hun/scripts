#! /usr/bin/env python3

import argparse
import json
import numpy as np
import pandas as pd

import detector_benchmark.lib as lib
import detector_benchmark.detection_parser as pair


def open_dataset(filename):
    with open(filename, "rb") as fp:
        return json.load(fp)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_path")
    parser.add_argument("query_path")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--skeleton", default=False, action="store_true")
    parser.add_argument("--output_name", default="measurements.csv")
    return parser


def convert_skeleton_to_bb_format(skeleton_file):
    boxes = []
    scores = []
    for skel in skeleton_file["skeletons"]:
        points = np.array([(pt["x"], pt["y"]) for pt in skel])
        bb = np.hstack((points.min(axis=0), points.max(axis=0)))
        boxes.append(list(bb))
        scores.append(np.array([pt["confidence"] for pt in skel]).mean())
    return {
        "image_paths": [skeleton_file["filename"]],
        "boxes": boxes,
        "labels_or_scores": scores,
        "skeletons": skeleton_file["skeletons"],
    }


def skeleton_cost(detection, detection_type, annotation_box, annotation_type):
    assert detection_type == "skeleton" and annotation_type == "body"
    points = np.array([(pt["x"], pt["y"]) for pt in detection])
    inside = (points[:, 0] >= annotation_box[0]) \
        & (points[:, 0] <= annotation_box[2]) \
        & (points[:, 1] >= annotation_box[1]) \
        & (points[:, 1] <= annotation_box[3])
    return -float(inside.sum()) / len(detection)


def extract_detections_skeletons(annotationList, annotationType, detectionList, detectionType, scaleRange, scoreThreshold,
                                 maskOutliers=False):
    gtCount = 0
    detections = []
    for index in range(len(annotationList)):
        annotationLabels = annotationList[index]["labels_or_scores"]
        annotationBoxes = annotationList[index]["boxes"]
        detectionScores = detectionList[index]["labels_or_scores"]
        detectionSkeletons = detectionList[index]["skeletons"]

        assert annotationList[index]["image_paths"] == detectionList[index]["image_paths"], \
            "{} != {}".format(annotationList[index]["image_paths"], detectionList[index]["image_paths"])

        annotationLabels = lib.annotationFilter(
            boxes=annotationBoxes,
            labels=annotationLabels,
            minHeight=scaleRange[0],
            maxHeight=scaleRange[1],
            maskOutliers=maskOutliers
        )
        positiveCount = len([label for label in annotationLabels if label not in lib.ignoredObjectLabels])
        gtCount += positiveCount

        detection = lib.max_pair(
            detectionSkeletons,
            detectionScores,
            detectionType,
            annotationBoxes,
            annotationLabels,
            annotationType,
            -scoreThreshold,
            minHeight=scaleRange[0],
            maxHeight=scaleRange[1],
            cost_fun=skeleton_cost,
        )
        detections.extend(detection)
    return detections, gtCount


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    gt_dataset = lib.readList(args.gt_path, count=1)
    query_dataset = open_dataset(args.query_path)
    query_filename_to_ind = {f["filename"]: ind for ind, f in enumerate(query_dataset)}

    query_bbox_dataset = [convert_skeleton_to_bb_format(query_dataset[query_filename_to_ind[gt_file["image_paths"][0]]])
                          for gt_file in gt_dataset]
    if args.skeleton:
        stats = pair.eval_scales(gt_dataset, "body", query_bbox_dataset, "skeleton", args.score_threshold,
                                 testNamePrefix="test1", extract_detections_fun=extract_detections_skeletons)["all"]
    else:
        stats = pair.eval_scales(gt_dataset, "body", query_bbox_dataset, "body", args.score_threshold,
                                 testNamePrefix="test1")["all"]

    pd.DataFrame(stats).to_csv(args.output_name, index=False, header=False, sep="\t")


if __name__ == "__main__":
    main()
