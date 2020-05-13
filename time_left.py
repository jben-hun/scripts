#! /usr/bin/python3

import sys
from datetime import datetime
from datetime import timedelta
# import statistics

if (len(sys.argv) < 3):
    sys.exit("usage: timeleft.py path steplimit")

path = sys.argv[1]
iterLimit = int(sys.argv[2])

times = []
speeds = []
with open(path, "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].rstrip()

    if "Iteration 0" in line:
        continue

    if i > 0 and "Test net" in lines[i-1].rstrip():
        continue

    if i > 0 and "prefetch queue empty" in lines[i-1].rstrip():
        continue

    found = line.find(" iter/s")
    if found != -1:
        time = line[1:21]
        time = datetime.strptime(time, "%m%d %H:%M:%S.%f")

        end = found
        start = line.rfind('(', 0, end) + 1
        speed = float(line[start:end])

        times.append(time)
        speeds.append(speed)

steps = min(20, len(speeds))
speed = sum(speeds[-steps:]) / steps
# speed = statistics.median(speeds)
# speed = sum(speeds)/len(speeds)
# speed = speeds[-1]

trainedFor = times[-1] - times[0]
stepsLeft = iterLimit - len(times)*50
timeLeft = stepsLeft / speed
timeLeft = timedelta(seconds=timeLeft)
completionDate = datetime.today() + timeLeft
print("trainedFor:     {}".format(trainedFor))
print("timeLeft:       {}".format(timeLeft))
print("completionDate: {}".format(completionDate))
print("currentDate:    {}".format(completionDate - timeLeft))
