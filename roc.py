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

COLORS = (
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
)

EPSILON = 1e-2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", default=".")
    parser.add_argument("-o", "--outfile", default=None)
    parser.add_argument("-p", "--precision", action="store_true")
    parser.add_argument("-s", "--scales", action="store_true")
    parser.add_argument("-c", "--confidence", action="store_true")
    args = parser.parse_args()

    precision  = args.precision
    confidence = args.confidence
    scales     = args.scales

    contents = os.listdir(args.indir)
    files = [ path.join(args.indir,file) for file in contents ]
    files = filter(path.isfile,files)
    files = [ file for file in files if path.splitext(file)[1] == ".csv" ]
    files.sort()

    assert files, "no tests to show"

    fig = go.Figure(layout=go.Layout(
        height=None,plot_bgcolor="white",
        xaxis_title="recall" if precision else "fp",
        yaxis_title="precision"+(" & confidence" if confidence else "") if precision else "recall"+(" & confidence" if confidence else ""),
        xaxis={"range":[0-EPSILON,1+EPSILON] if precision else None,"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
        yaxis={"range":[0-EPSILON,1+EPSILON],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
        legend={"tracegroupgap":15}
    ))

    colors = list(COLORS)
    # random.shuffle(colors)
    colorIterator = cycle(colors)
    netColors = {}
    recallMaxes = {}
    for i, file in enumerate(files):
        df = pd.read_csv(file,sep='\t',header=None,names=["recall","fp","confidence","precision"])

        test = path.basename(path.splitext(file)[0])
        specified,net,set,skip = parseName(test,scales)
        if skip:
            continue

        if set in recallMaxes:
            recallMaxes[set] = min( recallMaxes[set], df["recall"].max() )
        else:
            recallMaxes[set] = df["recall"].max()

    for i, file in enumerate(files):
        df = pd.read_csv(file,sep='\t',header=None,names=["recall","fp","confidence","precision"])

        test = path.basename(path.splitext(file)[0])
        specified,net,set,skip = parseName(test,scales)
        if skip:
            continue

        if precision:
            df = df[ df["recall"] <= recallMaxes[set] ]

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

            df = df.iloc[::-1]

            x = df["recall"].values
            y = df["precision"].values
            x_ = np.linspace(0,1,num=11)

            # f = interpolate.interp1d(x, y, fill_value="extrapolate")
            # y_ = f(x_)
            y_ = np.interp(x_, x, y, right=0)
            averagePrecision = np.mean(y_,dtype=np.float64)

            areaUnderCurve = np.trapz(df["precision"].values,df["recall"].values)

        else:
            df = df.groupby(["fp"],as_index=False,sort=False)
            df = df.agg({"recall":np.max,"confidence":np.min,"precision":np.max})

        if net not in netColors:
            color = next(colorIterator)
            netColors[net] = color
        else:
            color = netColors[net]

        hovertemplate = "<b>set:</b> {:}<br><b>net:</b> {:}<br>".format(set,net)+("<b>recall</b>" if precision else "<b>fp</b>")+" %{x}<br>"+("<b>precision</b>" if precision else "<b>recall</b>")+": %{y:.3f}<br><b>confidence</b> %{text:.3f}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=df["recall"] if precision else df["fp"],
            y=df["precision"] if precision else df["recall"],
            legendgroup=set if specified else None,
            name=("<b>auc/ap:</b> {:.3f}/{:.3f} ".format(areaUnderCurve,averagePrecision) if precision else "") + ("<b>set:</b> {:} <b>net:</b> {:}".format(set,net) if specified else "<b>{:}</b>".format(test)),
            mode="lines",
            hovertemplate=hovertemplate,
            text=df["confidence"],
            line={"color":color,"width":1.5},
            line_shape="linear"
        ))

        if confidence:
            fig.add_trace(go.Scatter(
                x=df["recall"] if precision else df["fp"],
                y=df["confidence"],
                legendgroup=set if specified else None,
                #name="<b>set:</b> {:} <b>net:</b> {:} confidence: {:}".format(set,net,confidence) if specified else "<b>{:}</b> confidence: {:}".format(test,confidence),
                name="confidence",
                mode="lines",
                line={"color":color,"width":1.5,"dash":"dash"},
                line_shape="linear"
            ))

    if precision:
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[1,0],
            mode="lines",
            name="guide1",
            line={"color":"black","width":1,"dash":"dash"},
            line_shape="linear"
        ))
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            name="guide2",
            line={"color":"black","width":1,"dash":"dash"},
            line_shape="linear"
        ))

    if args.outfile is None:
        outName = path.basename( os.getcwd() )
    else:
        outName = args.outfile

    outName += ("_pr" if precision else "_roc")
    outName += ("_scales" if scales else "")
    outName += ("_confidence" if confidence else "")
    outName += "_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
    outName += ".html"

    py.offline.plot(fig,filename=outName)

def parseName(test,scales):
    skip = False

    if "net" in test and "set" in test:
        specified = True
        testSplit = test.split(".")
        test = testSplit[0]
        net = test.split("net_")[1].split("_set_")[0]
        set = test.split("set_")[1].split("_net_")[0]
        if len(testSplit) > 1:
            if scales:
                if testSplit[1] == "all":
                    skip = True
                set += "_" + (".".join(testSplit[1:])).upper()
            elif testSplit[1] != "all":
                skip = True

    else:
        specified = False
        net = test
        set = test

    return specified, net, set, skip


if __name__ == '__main__':
    main()
