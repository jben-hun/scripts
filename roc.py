#!/usr/bin/python3

import plotly.graph_objects as go
import plotly as py
import pandas as pd
from sys import exit
import os, argparse, random
from itertools import cycle
import scipy.signal
import numpy as np
from scipy import interpolate

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--precision", action='store_true')
    parser.add_argument("-s", "--scales", action='store_true')
    parser.add_argument("-c", "--confidence", action='store_true')
    args = parser.parse_args()

    precision  = args.precision
    confidence = args.confidence

    contents = os.listdir()
    files = filter(os.path.isfile,contents)
    files = [ file for file in files if os.path.splitext(file)[1] == ".csv" ]
    files.sort()

    assert files, "no tests to show"

    fig = go.Figure(layout=go.Layout(
        height=None,plot_bgcolor="white",
        xaxis_title="recall" if precision else "fp",
        yaxis_title="precision" if precision else "recall",
        xaxis={"range":[0,1] if precision else None,"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
        yaxis={"range":[0,1],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1}
    ))

    colors = list(COLORS)
    # random.shuffle(colors)
    colorIterator = cycle(colors)
    netColors = {}
    for i, file in enumerate(files):
        df = pd.read_csv(file,sep='\t',header=None,names=["recall","fp","confidence","precision"])

        if precision:
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

        test = os.path.splitext(file)[0]
        if "net" in test and "set" in test:
            specified = True
            testSplit = test.split(".")
            test = testSplit[0]
            net = test.split("net_")[1].split("_set_")[0]
            set = test.split("set_")[1].split("_net_")[0]
            if len(testSplit) > 1:
                if args.scales:
                    if testSplit[1] == "all":
                        continue
                    set += "_" + (".".join(testSplit[1:])).upper()
                elif testSplit[1] != "all":
                    continue

        else:
            specified = False
            net = test
            set = test

        if net not in netColors:
            color = next(colorIterator)
            netColors[net] = color
        else:
            color = netColors[net]

        hovertemplate = "<b>set "+set+"<br>net "+net+"</b><br>"+("recall" if precision else "fp")+" %{x}<br>"+("precision" if precision else "recall")+": %{y:.3f}<br>confidence %{text:.3f}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=df["recall"] if precision else df["fp"],
            y=df["precision"] if precision else df["recall"],
            legendgroup=set if specified else None,
            name=("AUC {:.3f} AP {:.3f} ".format(areaUnderCurve,averagePrecision) if precision else "")+"set "+set+" net "+net if specified else test,
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
                name="set: "+set+" net: "+net+" confidence" if specified else test+" confidence",
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

    py.offline.plot(fig, filename=os.path.basename( os.getcwd() ) + ".html")

def isMonotonic(A):

    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


if __name__ == '__main__':
    main()
