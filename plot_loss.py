#! /usr/bin/python3

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt, numpy as np, sys, os, argparse, scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument("logfile")
parser.add_argument("-w", "--window", default=1, type=int)

args = parser.parse_args()

path = args.logfile

trainWindow = args.window * 100
testWindow = args.window * 1

fig, ax = plt.subplots()

with open(path, "r") as f:
    lines = f.readlines()

trainIterations = []
trainLossesCls = []
trainLossesReg = []

testIterations = []
testLossesCls = []
testLossesReg = []

pattern = "sgd_solver.cpp"
filteredLines = filter(lambda x: pattern not in x, lines)

# pattern = " iters), loss = "
# filteredLines = filter(lambda x: pattern in x, lines)
# for line in filteredLines:
#     line = line.rstrip()
#
#     start = line.find(pattern)+len(pattern)
#     loss = float(line[start:])
#     losses.append(loss)
#     iterations.append(float( line.split("Iteration")[1].split(' ')[1] ))

pattern = " iters), loss = "
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    trainIterations.append(int( line.split("Iteration ")[1].split(' ')[0] ))

pattern = "Train net output #1"
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    trainLossesCls.append(float( line.split("rpn_cls_loss = ")[1].split(' ')[0] ))

pattern = "Train net output #2"
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    trainLossesReg.append(float( line.split("rpn_loss_bbox = ")[1].split(' ')[0] ))

pattern = "Testing net (#0)"
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    testIterations.append(int( line.split("Iteration ")[1].split(' ')[0][:-1] ))

pattern = "Test net output #0"
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    testLossesCls.append(float( line.split("rpn_cls_loss = ")[1].split(' ')[0] ))

pattern = "Test net output #1"
filteredLines = filter(lambda x: pattern in x, lines)
for line in filteredLines:
    testLossesReg.append(float( line.split("rpn_loss_bbox = ")[1].split(' ')[0] ))

# plot train losses
y1 = np.array(trainLossesCls)
y2 = np.array(trainLossesReg)
x = np.array(trainIterations)
length = min([ x.shape[0], y1.shape[0], y2.shape[0] ])
if trainWindow != 1:
    y1 = np.convolve(y1, np.ones((trainWindow,))/trainWindow, mode="same")
    y2 = np.convolve(y2, np.ones((trainWindow,))/trainWindow, mode="same")

plt.plot(x[:length],y1[:length],label="train cls")
plt.plot(x[:length],y2[:length],label="train reg")

# plot test losses
y1 = np.array(testLossesCls)
y2 = np.array(testLossesReg)
x = np.array(testIterations)
length = min([ x.shape[0], y1.shape[0], y2.shape[0] ])
if testWindow != 1:
    y1 = np.convolve(y1, np.ones((testWindow,))/testWindow, mode="same")
    y2 = np.convolve(y2, np.ones((testWindow,))/testWindow, mode="same")

plt.plot(x[:length],y1[:length],label="test cls")
plt.plot(x[:length],y2[:length],label="test reg")

ax.set(xlabel='iterations', ylabel='losses')
ax.legend()
ax.grid()

#fig.savefig("losses.png")
plt.show()
