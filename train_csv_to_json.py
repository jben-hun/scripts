#!/usr/bin/python3

from shutil import copytree, copy2
from pprint import pprint
from os import path
from sys import exit, argv
import json, datetime, argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("-o","--output_dir",default="data")
    parser.add_argument("-a","--annotator",default="head_det v3 0_7")
    args = parser.parse_args()

    assert not path.exists(args.output_dir)

    taskDirs = set()
    train = {}
    with open(args.csv_file,'r') as f:
        for line in f:
            splitLine = line.rstrip().split('\t')

            src = splitLine[0]
            dst = path.join(args.output_dir,path.dirname(src).lstrip(os.sep))
            boxes = splitLine[1:]
            boxes = [ boxes[i] for i in range(len(boxes)) if (i+1)%5 != 0 ]

            d = {}
            d["image"] = path.basename(src)

            objects = []
            i = 0
            while i < len(boxes):
                x1 = int(boxes[i])
                y1 = int(boxes[i+1])
                x2 = int(boxes[i+2])
                y2 = int(boxes[i+3])
                i+=4

                objects.append(
                    {
                         "name": "person",
                         "head": {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
                    }
                )

            os.makedirs(dst, exist_ok=True)
            copy2(src, dst)

            d["annotations"] = [
                {
                    "annotator": args.annotator,
                    "date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "objects" : objects
                }
            ]

            jsonPath = path.join(dst, (path.basename(src))+".json")

            with open(jsonPath, 'w') as outfile:
                json.dump(d, outfile, indent=2)

            taskDirs.add(dst)


    taskDirs = list(taskDirs)
    taskDirs = [
        {
            "id": i+1,
            "dir": taskDirs[i]+os.sep
        }
        for i in range(len(taskDirs))
    ]

    with open( path.join(args.output_dir,"tasks.json"), 'w' ) as f:
        json.dump(taskDirs, f, indent=2)

if __name__ == "__main__":
    main()
