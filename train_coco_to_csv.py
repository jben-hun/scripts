#! /usr/bin/python3

import argparse
import json
import os
import shutil
from os import sys
from os import path
from pprint import pprint
from sys import exit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-m","--mask_crowd",action="store_true")
    args = parser.parse_args()

    assert path.exists(args.json_file)

    imageIdToName = {}
    annotationsPerImage = {}
    with open(args.json_file) as f:
        d = json.load(f)
        for image in d["images"]:
            imageIdToName[image["id"]] = image["file_name"]
            annotationsPerImage[image["file_name"]] = []

        for annot in d["annotations"]:
            if annot["category_id"] == 1:
                image = imageIdToName[annot["image_id"]]
                bbox = [
                    annot["bbox"][0],
                    annot["bbox"][1],
                    annot["bbox"][0]+annot["bbox"][2],
                    annot["bbox"][1]+annot["bbox"][3]
                ]
                label = 0 if args.mask_crowd and annot["iscrowd"] else 1

                annotationsPerImage[image] += bbox + [label]
            else:
                print("non-human annotation found")

    with open(path.splitext(args.json_file)[0]+("_masked_crowds" if args.mask_crowd else "")+".csv","w") as f:
        for image,detections in annotationsPerImage.items():
            if detections:
                line = [image]+detections
                line = list(map(str,line))
                line = "\t".join(line)
                f.write(line+"\n")



if __name__ == '__main__':
    main()
