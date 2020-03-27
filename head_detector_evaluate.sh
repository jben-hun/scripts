#!/bin/bash

if [[ ! $model ]] || [[ ! $annotations ]]
then
  echo \"model\" and \"annotations\" are mandatory parameters
  exit 1
fi
detections="${detections-/tmp/head_detector_evaluation_detections.csv}"
score_threshold="${score_threshold-0.01}"
count="${count-1}"
dldemo_exe="${dldemo_exe-$HOME/dldemo}"
dldemo_cmd="${dldemo_cmd-rpndet}"
min_height="${min_height-1}"
max_height="${max_height-1000}"
bb_match_metric="${bb_match_metric-IOU}"
bb_match_threshold="${bb_match_threshold-0.5}"
square_annotations="${square_annotations-0}"
case $square_annotations in
  0) square_annotations="" ;;
  *) square_annotations="--square_annotations" ;;
esac
square_detections="${square_detections-0}"
case $square_detections in
  0) square_detections="" ;;
  *) square_detections="--square_detections" ;;
esac
verbose="${verbose-0}"
case $verbose in
  0) silent=1 ;;
  *) silent=0 ;;
esac

"$dldemo_exe" \
"$dldemo_cmd" \
"$model" \
"$annotations" \
"$detections" \
"$min_height" \
"$max_height" \
"$score_threshold" \
"$count" \
"$silent"

IFS="/"
arr_set=($annotations)
arr_net=($model)
IFS="."
arr_set=(${arr_set[-1]})
unset IFS
set="${arr_set[0]}"
net="${arr_net[-2]}"
name="$HOME/detector_evaluation_results/set_${set}_net_${net}.csv"
output="${output-$name}"
dir=`dirname "$output"`
mkdir -p "$dir"
dir=`dirname "$0"`
python3 "$dir/detector_evaluation/pr_curve_compute.py" \
--annotations="$annotations" \
--detections="$detections" \
--output="$output" \
--bb_match_metric="$bb_match_metric" \
--bb_match_threshold="$bb_match_threshold" \
$square_detections \
$square_annotations
