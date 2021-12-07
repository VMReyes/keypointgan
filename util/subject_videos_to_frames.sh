#!/bin/bash

VIDEOS_PATH="/data/vision/torralba/scratch/vmreyes/keypointgan/datasets/simple_human36m/human_images/S5/Videos/"
VIDEOS="/data/vision/torralba/scratch/vmreyes/keypointgan/datasets/simple_human36m/human_images/S5/Videos/*"
detox -r $VIDEOS
rm $VIDEOS_PATH/*ALL*
VIDEO_OUT="/data/vision/torralba/scratch/vmreyes/keypointgan/datasets/simple_human36m/human_images/S5/Videos/"

for f in $VIDEOS
do
	echo "Processing $f video..."
	filename_with_path_activity_and_id="${f%.*}"
	filename_no_path=$(basename -- "$f")
	filename_no_path_with_id="${filename_no_path%*.*}"
	echo "filename with path activity and id: $filename_with_path_activity_and_id"
	echo "filename with no path: $filename_no_path"
	echo "filename with no path with id: $filename_no_path_with_id"
	mkdir $filename_with_path_activity_and_id
	ffmpeg -i $f -vf fps=50 $filename_with_path_activity_and_id/frame-%06d.png
done

