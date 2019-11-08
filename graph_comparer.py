#!/usr/bin/ipython

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.offline.offline import _plot_html
from plotly.graph_objs import *
# Create random data with numpy
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compare n detector report')
parser.add_argument('reports', type=str, nargs='+',
                   help='File names for reports')

args = parser.parse_args()


init_notebook_mode()

data = []

deploy_path='DEPLOY/'
for i in args.reports:
  data_x = []
  data_y = []
  with open(i, 'r') as f:
    count = 0
    for line in f:
        count+=1
        if count % 20 == 0 and line!='\n' and line!='\r\n' and len(line)!=0:
          line_block = [float(x) for x in line.split()]
          if not len(line_block) == 0 :
            data_x.append(line_block[1])
            data_y.append(line_block[0])
  desc = ""
  trace = Scatter(
    x = data_x,
    y = data_y,
    mode = 'lines',
    name = 'Experiment '+str(i),
    text = desc
  )
  data.append(trace)

# Create traces

layout = Layout(
    showlegend=True,
    height=1000,
    width=1800,
    hovermode='closest',
    xaxis=dict(
        title="False positive (hh100%={}, yt100%={})".format(2649,4254),
        range=[0,2000],
    ),
    yaxis=dict(
        title="True positive rate",
        range=[0,1],
    )
)
# Plot and embed in ipython notebook!
fig= dict(data=data, layout=layout)
#df=iplot(fig)

# plot_html, plotdivid, width, height = _plot_html(fig, False, "", True, '100%', 525,'')
# _plot_html(figure_or_data, config, validate, default_width, default_height, global_requirejs)
plot_html, plotdivid, width, height = _plot_html(fig, "", True, '100%', 525,'')

html_start = """
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>"""

html_end = """
</body>
</html>"""

html_final = html_start + plot_html + html_end
f = open("compare.html", 'w')
f.write(html_final)
f.close()
