#!/usr/bin/python3

import plotly.graph_objects as go
import plotly as py
import pandas as pd
from sys import exit
import os, argparse, random
from itertools import cycle
import scipy.signal

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
    args = parser.parse_args()

    precision = args.precision

    contents = os.listdir()
    files = filter(os.path.isfile,contents)
    files = [ file for file in files if os.path.splitext(file)[1] == ".csv" ]
    files.sort()

    assert files, "no tests to show"

    fig = go.Figure(layout=go.Layout(
        height=None,plot_bgcolor="white",
        xaxis_title="precision" if precision else "fp",
        yaxis_title="recall",
        xaxis={"range":[0,1] if precision else None,"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1},
        yaxis={"range":[0,1],"gridcolor":"lightGray","gridwidth":1,"zerolinecolor":"black","zerolinewidth":1}
    ))

    colors = list(COLORS)
    random.shuffle(colors)
    colorIterator = cycle(colors)
    netColors = {}
    for i, file in enumerate(files):
        df = pd.read_csv(file,sep='\t',header=None,names=["recall","fp","confidence","precision"])

        if precision:
            df = df.sort_values(by=["precision"],ascending=False)
            maxRecall = 0
            for index, row in df.iterrows():
                maxRecall = max(maxRecall,row["recall"])
                df.at[index,"recall"] = maxRecall

        test = os.path.splitext(file)[0]
        if "net" in test and "set" in test:
            specified = True
            net = test.split("net_")[1].split("_set_")[0]
            set = test.split("set_")[1].split("_net_")[0]
        else:
            specified = False
            net = test
            set = test

        if net not in netColors:
            color = next(colorIterator)
            netColors[net] = color
        else:
            color = netColors[net]

        if precision:
            hovertemplate = "<b>set: "+set+"<br>net: "+net+"</b><br>precision: %{x:.3f}<br>recall: %{y:.3f}<br>confidence: %{text:.3f}<extra></extra>"
        else:
            hovertemplate = "<b>set: "+set+"<br>net: "+net+"</b><br>fp: %{x}<br>recall: %{y:.3f}<br>confidence: %{text:.3f}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=df["precision"] if precision else df["fp"],
            y=df["recall"],
            legendgroup=set if specified else None,
            name="set: "+set+" net: "+net if specified else test,
            mode="lines",
            hovertemplate=hovertemplate,
            text=df["confidence"],
            line={"color":color,"width":1.5}
        ))

    if precision:
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[1,0],
            mode="lines",
            name="guide",
            line={"color":"black","width":1,"dash":"dash"}
        ))

    py.offline.plot(fig, filename=os.path.basename( os.getcwd() ) + ".html")



if __name__ == '__main__':
    main()
