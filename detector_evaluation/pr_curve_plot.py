#!/usr/bin/python3

import argparse
import datetime
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objects as go
from os import path

EPSILON = 1e-2
NAME_CHAR_LIMIT = 70
LEGEND_WIDTH = 500

def main():
    args = parseArguments()

    files = sorted([ file for file in args.input if isCsvFile(file) ])

    assert files, "no valid tests available to show"

    fig = createFigure(
        xaxis_title="recall",
        yaxis_title="precision"+(" & score" if args.score else ""),
        xaxis={"range":(0-EPSILON,1+EPSILON),"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
        yaxis={"range":(0-EPSILON,1+EPSILON),"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1}
    )

    for file in files:
        df = pd.read_csv(file,sep=unescape(args.delimiter),header=None,names=("score","precision","recall"))

        name = path.basename(path.splitext(file)[0])
        addScatterTrace(
            fig=fig,
            x=df["recall"],
            y=df["precision"],
            name=name,
            hovertemplate="<b>"+name+"</b><br><b>recall</b>: %{x}<br><b>precision</b>: %{y}<br><b>score</b>: %{text}<extra></extra>",
            text=df["score"]
        )

        if args.score:
            addScatterTrace(
                fig=fig,
                x=df["recall"],
                y=df["score"],
                name="score",
                dash="dash"
            )

    addScatterTrace(
        fig=fig,
        x=(0,1),
        y=(1,0),
        name="guide1",
        color="black",
        dash="dashdot"
    )

    addScatterTrace(
        fig=fig,
        x=(0,1),
        y=(0,1),
        name="guide2",
        color="black",
        dash="dashdot"
    )

    py.offline.plot(fig,filename=args.output, auto_open=False)

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
        margin={"r":LEGEND_WIDTH,"autoexpand":False}
    ))

def addScatterTrace(fig,x,y,name,hovertemplate=None,text=None,color=None,dash=None):
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name=cutLonger(name,NAME_CHAR_LIMIT),
        mode="lines",
        hovertemplate=hovertemplate,
        line={"color":color,"width":1.5,"dash":dash},
        line_shape="linear",
        text=text
    ))

def unescape(s):
    return bytes(s, "utf-8").decode("unicode_escape")

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", nargs="+", required=True,)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-s", "--score", action="store_true")
    parser.add_argument("-d", "--delimiter", default="\t")

    return parser.parse_args()

def cutLonger(s,n):
    return s if len(s) <= n else s[:n//2]+"~"+s[-n//2:]



if __name__ == '__main__':
    main()
