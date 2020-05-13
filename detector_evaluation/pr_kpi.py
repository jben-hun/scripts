#!/usr/bin/python3

import os
import argparse
import pandas as pd
from os import path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("-o", "--outfile")
    parser.add_argument(
        "-p", "--precisions", nargs='+',
        default=[0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8])
    parser.add_argument("--delimiter", default="\t")
    args = parser.parse_args()

    assert path.isfile(args.infile)

    df = pd.read_csv(
        args.infile, sep=unescape(args.delimiter), header=None,
        names=("score", "precision", "recall"))

    if args.outfile is None:
        outfile = path.basename(args.infile)
        outfile = path.join("kpi", outfile)
    else:
        outfile = args.outfile
        outfile = path.join("kpi", outfile)

    header = ["target_precision", "precision", "recall", "score"]
    rows = [header]

    recalls = []
    for target_precision in args.precisions:
        argMin = np.abs(df["precision"].values - target_precision).argmin()

        precision = df["precision"].values[argMin]
        recall = df["recall"].values[argMin]
        score = df["score"].values[argMin]

        row = [target_precision, precision, recall, score]
        rows.append(row)
        recalls.append(recall)

    averageRecall = np.mean(recalls, dtype=np.float64)

    os.makedirs(path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        for row in rows:
            row = [str(e) for e in row]
            line = ",".join(row)
            f.write(line+"\n")
        f.write("average_recall"+"\n")
        f.write(str(averageRecall)+"\n")


def unescape(s):
    return bytes(s, "utf-8").decode("unicode_escape")


if __name__ == '__main__':
    main()
