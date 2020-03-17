#!/usr/bin/python3

"""
This script is meant to produce various performance curves of convolutional
neural networks. It reads all .csv files from a specified folder, plots a curve,
and writes it to an interactive .html file.

Each .csv file is a result of a single model evaluated on a dataset.

The naming of a .csv should include the following two elements, in any order:
- The "net_" keyword, followed by a string identifying the neural network model.
- The "set_" keywork, followed by a string identifying the dataset.

The column structure of the .csv files should be:
- recall, fp, confidence, precision
in this order, where fp denotes the ammount of false positives.

See detector_benchmark/pair.py
"""

import os
import argparse
import random
import datetime
import scipy.signal
import textwrap
import plotly.graph_objects as go
import plotly as py
import pandas as pd
import numpy as np
from sys import exit
from os import path
from itertools import cycle
from scipy import interpolate
from pprint import pprint

COLORS = (
    '#1f77b4', # muted blue
    '#ff7f0e', # safety orange
    '#2ca02c', # cooked asparagus green
    '#d62728', # brick red
    '#9467bd', # muted purple
    '#8c564b', # chestnut brown
    '#e377c2', # raspberry yogurt pink
    '#7f7f7f', # middle gray
    '#bcbd22', # curry yellow-green
    '#17becf'  # blue-teal
)

EPSILON = 1e-2

def main():
    args = parseArguments()

    contents = os.listdir(args.indir)
    contents = [ path.join(args.indir,content) for content in contents ]
    files = [ content for content in contents if isCsvFile(content) ]
    files.sort()

    assert files, "no tests to show"

    colors = list(COLORS)
    if args.random_color:
        random.shuffle(colors)
    colorIterator = cycle(colors)
    netColors = {}

    if args.precision: # draw precision-recall curves
        fig = createFigure(
            xaxis_title="recall",
            yaxis_title="precision"+(" & confidence" if args.confidence else ""),
            xaxis={"range":[0-EPSILON,1+EPSILON],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
            yaxis={"range":[0-EPSILON,1+EPSILON],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1}
        )

        recallMaxes = {}
        for i, file in enumerate(files):
            df = pd.read_csv(file,sep=unescape(args.delimiter),header=None,names=["recall","fp","confidence","precision"])

            test = path.basename(path.splitext(file)[0])
            specified,net,set,skip = parseName(test,args.scales)
            if skip:
                continue

            if set in recallMaxes:
                recallMaxes[set] = min( recallMaxes[set], df["recall"].max() )
            else:
                recallMaxes[set] = df["recall"].max()

        for i, file in enumerate(files):
            df = pd.read_csv(file,sep=unescape(args.delimiter),header=None,names=["recall","fp","confidence","precision"])

            test = path.basename(path.splitext(file)[0])
            specified,net,set,skip = parseName(test,args.scales)
            if skip:
                continue

            df = df[ df["recall"] <= recallMaxes[set] ]

            df = (
                df.groupby(["recall"],as_index=False,sort=False)
                .agg({"fp":np.min,"confidence":np.max,"precision":np.max})
                .sort_values(by=["recall"],ascending=False)
            )

            df["precision"] = df["precision"].cummax()
            df["confidence"] = df["confidence"].cummax()

            df = df.iloc[::-1]

            x = df["recall"].to_numpy()
            y = df["precision"].to_numpy()
            x_ = np.linspace(0,1,num=11)

            # f = interpolate.interp1d(x, y, fill_value="extrapolate")
            # y_ = f(x_)
            y_ = np.interp(x_, x, y, right=0)
            averagePrecision = np.mean(y_,dtype=np.float64)

            areaUnderCurve = np.trapz(y,x)

            if net not in netColors:
                color = next(colorIterator)
                netColors[net] = color
            else:
                color = netColors[net]

            addScatterTrace(
                fig=fig,
                x=df["recall"],
                y=df["precision"],
                legendgroup=set if specified else None,
                name=(
                      "<b>auc/ap:</b> {:.3f}/{:.3f} ".format(areaUnderCurve,averagePrecision)
                    + ("<b>set:</b> {:} <b>net:</b> {:}".format(cutLonger(set),cutLonger(net)) if specified else "<b>{:}</b>".format(test))
                ),
                hovertemplate=(
                      "<b>set:</b> {:}<br><b>net:</b> {:}<br>".format(set,net)
                    + "<b>recall</b> %{x}<br><b>precision</b>: %{y:.3f}<br><b>confidence</b> %{text:.3f}<extra></extra>"
                ),
                text=df["confidence"],
                color=color,
                dash=None
            )

            if args.confidence:
                addScatterTrace(
                    fig=fig,
                    x=df["recall"],
                    y=df["confidence"],
                    legendgroup=set if specified else None,
                    name="confidence",
                    hovertemplate=None,
                    text=None,
                    color=color,
                    dash="dash"
                )

        addScatterTrace(
            fig=fig,
            x=[0,1],
            y=[1,0],
            legendgroup=None,
            name="guide1",
            hovertemplate=None,
            text=None,
            color="black",
            dash="dashdot"
        )

        addScatterTrace(
            fig=fig,
            x=[0,1],
            y=[0,1],
            legendgroup=None,
            name="guide2",
            hovertemplate=None,
            text=None,
            color="black",
            dash="dashdot"
        )

    else: # draw ROC curves
        fig = createFigure(
            xaxis_title="fp",
            yaxis_title="recall"+(" & confidence" if args.confidence else ""),
            xaxis={"range":None,"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
            yaxis={"range":[0-EPSILON,1+EPSILON],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1}
        )

        for i, file in enumerate(files):
            df = pd.read_csv(file,sep=unescape(args.delimiter),header=None,names=["recall","fp","confidence","precision"])

            test = path.basename(path.splitext(file)[0])
            specified,net,set,skip = parseName(test,args.scales)
            if skip:
                continue

            df = (
                df.groupby(["fp"],as_index=False,sort=False)
                .agg({"recall":np.max,"confidence":np.min,"precision":np.max})
            )

            if net not in netColors:
                color = next(colorIterator)
                netColors[net] = color
            else:
                color = netColors[net]

            addScatterTrace(
                fig=fig,
                x=df["fp"],
                y=df["recall"],
                legendgroup=set if specified else None,
                name=(
                    ("<b>set:</b> {:} <b>net:</b> {:}".format(cutLonger(set),cutLonger(net)) if specified else "<b>{:}</b>".format(test))
                ),
                hovertemplate=(
                      "<b>set:</b> {:}<br><b>net:</b> {:}<br>".format(set,net)
                    + "<b>fp</b> %{x}<br><b>recall</b>: %{y:.3f}<br><b>confidence</b> %{text:.3f}<extra></extra>"
                ),
                text=df["confidence"],
                color=color,
                dash=None
            )

            if args.confidence:
                addScatterTrace(
                    fig=fig,
                    x=df["fp"],
                    y=df["confidence"],
                    legendgroup=set if specified else None,
                    name="confidence",
                    hovertemplate=None,
                    text=None,
                    color=color,
                    dash="dash"
                )

    if args.outfile is None:
        outName = path.basename( os.getcwd() )
    else:
        outName = args.outfile

    outName += ("_pr" if args.precision else "_roc")
    outName += ("_scales" if args.scales else "")
    outName += ("_confidence" if args.confidence else "")
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

def isCsvFile(file):
    return path.isfile(file) and path.splitext(file)[1] == ".csv"

def createFigure(xaxis_title,yaxis_title,xaxis,yaxis):
    return go.Figure(layout=go.Layout(
        height=None,
        plot_bgcolor="white",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=xaxis,
        yaxis=yaxis,
        legend={"tracegroupgap":15},
        margin={"r":500}
    ))

def addScatterTrace(fig,x,y,legendgroup,name,hovertemplate,text,color,dash):
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        legendgroup=legendgroup,
        name=name,
        mode="lines",
        hovertemplate=hovertemplate,
        line={"color":color,"width":1.5,"dash":dash},
        #line_shape="linear",
        text=text
    ))

def unescape(s):
    return bytes(s, "utf-8").decode("unicode_escape")

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--indir", default=".")
    parser.add_argument("-o", "--outfile", default=None)
    parser.add_argument("-p", "--precision", action="store_true")
    parser.add_argument("-s", "--scales", action="store_true")
    parser.add_argument("-c", "--confidence", action="store_true")
    parser.add_argument("-r", "--random_color", action="store_true")
    parser.add_argument("-d", "--delimiter",default="\t")

    return parser.parse_args()

def htmlWrap(s,width=15):
    return "<br>".join(textwrap.wrap(s,width=width))

def cutLonger(s,n=15):
    return s if len(s) <= n else s[:n//2]+"..."+s[-n//2:]



if __name__ == '__main__':
    main()
