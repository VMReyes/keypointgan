#!/bin/bash

VIDEOS="/home/vmreyes/keypointgan/datasets/simple_human36m/human_images/S1/Videos/*"
detox -r $VIDEOS
VIDEO_OUT="/home/vmreyes/keypointgan/datasets/simple_human36m/human_images/S1/Videos/"

for f in $VIDEOS
do
	echo "Processing $f video..."
	filename_with_path_activity_and_id="${f%.*}"
	filename_no_path=$(basename -- "$f")
	filename_no_path_with_id="${filename_no_path%*.*}"
	echo "$filename_with_path_activity_and_id"
	echo "$filename_no_path"
	echo "$filename_no_path_with_id"
	mkdir $filename_with_path_activity_and_id
	ffmpeg -i $f -vf fps=50 $filename_with_path_activity_and_id/frame-%06d.png
done

