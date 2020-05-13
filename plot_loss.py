#! /usr/bin/python3

import argparse
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

BLACKLIST = {"accuracy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("test_window_size", type=int)
    parser.add_argument("train_window_size", type=int)
    parser.add_argument("-m", "--median", action='store_true')
    parser.add_argument("-y", "--ylim", type=float)
    args = parser.parse_args()

    path = args.logfile
    trainWindow = args.train_window_size
    testWindow = args.test_window_size

    fig, ax = plt.subplots()

    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    testLines = filter(
        lambda x: "Testing net" in x or "Test net output" in x, lines)
    trainLines = filter(
        lambda x: " iters), loss = " in x or "Train net output" in x, lines)

    testLosses, trainLosses, testIterations, trainIterations = {}, {}, [], []

    for line in testLines:
        found = line.find("Iteration")
        if found != -1:
            testIterations.append(int(line[found:].split(" ")[1][:-1]))
        else:
            fields = line.split("Test net output")[1].split(" ")
            name = fields[2]
            if name not in BLACKLIST:
                value = float(fields[4])
                mapAppend(testLosses, name, value)

    for line in trainLines:
        found = line.find("Iteration")
        if found != -1:
            trainIterations.append(int(line[found:].split(" ")[1]))
        else:
            fields = line.split("Train net output")[1].split(" ")
            name = fields[2]
            if name not in BLACKLIST:
                value = float(fields[4])
                mapAppend(trainLosses, name, value)

    # plot train losses
    length = min(
        [len(v) for v in trainLosses.values()] + [len(trainIterations)])
    x = np.array(trainIterations)
    for name, losses in trainLosses.items():
        y = np.array(losses)
        if args.median:
            y = scipy.signal.medfilt(y, trainWindow)
        else:
            yPadded = np.pad(y, (trainWindow-1, 0), "edge")
            y = np.convolve(
                yPadded, np.ones((trainWindow,))/trainWindow, mode="valid")
        plt.plot(x[:length], y[:length], label="train "+name)

    # plot test losses
    length = min(
        [len(v) for v in testLosses.values()] + [len(testIterations)])
    x = np.array(testIterations)
    # padSize = (testWindow-1)//2, (testWindow-1) - (testWindow-1)//2
    for name, losses in testLosses.items():
        y = np.array(losses)
        if args.median:
            y = scipy.signal.medfilt(y, trainWindow)
        else:
            yPadded = np.pad(y, (testWindow-1, 0), "edge")
            y = np.convolve(
                yPadded, np.ones((testWindow,))/testWindow, mode="valid")
        plt.plot(x[:length], y[:length], label="test "+name)

    ax.set(xlabel='iterations', ylabel='losses')
    if args.ylim is not None:
        ax.set_ylim(bottom=0, top=args.ylim)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def mapAppend(map, key, value):
    if key in map:
        map[key].append(value)
    else:
        map[key] = [value]


if __name__ == '__main__':
    main()
