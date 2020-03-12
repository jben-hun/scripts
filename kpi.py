#!/usr/bin/python3

import plotly.graph_objects as go
import plotly as py
import pandas as pd
from sys import exit
import os, argparse, random, datetime
from os import path
from itertools import cycle
import scipy.signal
import numpy as np
from scipy import interpolate
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("-o", "--outfile", default=None)
    parser.add_argument("-p", "--precisions",nargs='+',default=[0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8])
    args = parser.parse_args()

    assert path.isfile(args.infile)

    df = pd.read_csv(args.infile,sep='\t',header=None,names=["recall","fp","confidence","precision"])

    if args.outfile is None:
        outfile = path.basename(args.infile)
        outfile = path.join("kpi",outfile)
    else:
        outfile = args.outfile + ".csv"
        outfile = path.join("kpi",outfile)

    df = df.groupby(["recall"],as_index=False,sort=False)
    df = df.agg({"fp":np.min,"confidence":np.max,"precision":np.max})
    df = df.sort_values(by=["recall"],ascending=False)

    maxPrecision = 0
    maxConfidence = 0
    for index, row in df.iterrows():
        maxPrecision = max(maxPrecision,row["precision"])
        maxConfidence = max(maxConfidence,row["confidence"])
        df.at[index,"precision"] = maxPrecision
        df.at[index,"confidence"] = maxConfidence

    header = [ "target_precision","precision","recall","confidence","fp" ]
    rows = [ header ]

    recalls = []
    for target_precision in args.precisions:
        argMin = np.abs(df["precision"].values - target_precision).argmin()

        precision  = df["precision"].values[argMin]
        recall     = df["recall"].values[argMin]
        confidence = df["confidence"].values[argMin]
        fp  = df["fp"].values[argMin]

        row = [ target_precision,precision,recall,confidence,fp ]
        rows.append(row)
        recalls.append(recall)

    averageRecall = np.mean(recalls,dtype=np.float64)

    os.makedirs( path.dirname(outfile), exist_ok=True )
    with open(outfile,"w") as f:
        for row in rows:
            row = [ str(e) for e in row ]
            line = ",".join(row)
            f.write(line+"\n")
        f.write("average_recall"+"\n")
        f.write(str(averageRecall)+"\n")



if __name__ == '__main__':
    main()
