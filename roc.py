#!/usr/bin/python3

import plotly as py
import plotly_express as px
import pandas as pd
from sys import exit
import os

contents = os.listdir()
files = filter(os.path.isfile,contents)
files = [ file for file in files if os.path.splitext(file)[1] == ".csv" ]
files.sort()

dataFrames = []
for i, file in enumerate(files):
    df = pd.read_csv(file,sep='\t',header=None,names=["recall","fp","confidence"])
    df["test"] = os.path.splitext(file)[0]
    dataFrames.append(df)

df = pd.concat(dataFrames)

fig = px.line(df, x="fp", y="recall", range_x=[0,2000], range_y=[0,1], color="test", height=700, line_shape="spline", render_mode="svg")
fig.update_layout(plot_bgcolor="white")
fig.update_xaxes(gridcolor='lightGray',gridwidth=1,zerolinecolor="black",zerolinewidth=1)
fig.update_yaxes(gridcolor='lightGray',gridwidth=1,zerolinecolor="black",zerolinewidth=1)

py.offline.plot(fig, filename=os.path.basename( os.getcwd() ) + ".html")
