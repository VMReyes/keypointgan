import os, sys
import mat73
import argparse
import skimage.io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simplifies a Human3.6m dataset after running subject_videos_to_frames.sh')
parser.add_argument('--root_dir', required=True,
                    help='Root directory of the Human3.6m dataset after converting videos to frames using subject_videos_to_frames.sh')

def bb_box_array_to_coords(bbox):
    """Returns the top left and bottom right coordinates of a bounding box
       matrix as provided by Humans3.6m.

    :param bbox: bounding box matrix with elements true or false representing
                 the bounding box around the subject
    :type bbox: np.array
    :rtype: list of 2
    :return: Top left, bottom right coordinates of the bounding box
    """
    top_left = None
    for x in range(len(bbox)):
        for y in range(len(bbox[0])):
            if bbox[x, y] == True:
                top_left = (x,y)
                break
        if top_left != None:
            break

    bottom_right = None
    for x in range(len(bbox)):
        for y in range(len(bbox[0])):
            if bbox[x, y] == True:
                bottom_right = (x,y)
    
    return top_left, bottom_right

def bb_box_to_square_bb(top_left, bottom_right):
    # Calculate height and width of bounding box
    height = bottom_right[0] - top_left[0]
    width = bottom_right[1] - top_left[1]
    if height > width:
        offset = int((height - width) / 2)
        new_top_left = (top_left[0], top_left[1] - offset)
        new_bottom_right = (bottom_right[0], bottom_right[1]+offset)
        return new_top_left, new_bottom_right
    else:
        print("TODO")
        assert False

if __name__ == '__main__':
    args = parser.parse_args()
    # Expecting actor folders in root directory
    subject_folders = next(os.walk(args.root_dir))[1]
    for subject in subject_folders:
        print(f'Processing subject: {subject}')
        # Expecting a "Videos" folder in every actor's folder
        videos_folder = os.path.join(args.root_dir, subject, "Videos")
        # Expecting a folder for every video from that subject, where the video has been
        # converted to its frames
        frame_by_frame_video_folders = next(os.walk(videos_folder))[1]
        frame_by_frame_video_folders.sort()

        for video in frame_by_frame_video_folders:
            print(f'Processing video: {video}')

            # Expecting a "SegmentsMat" folder in every actor's folder
            # Expecting a "ground_truth_bb" folder in every "SegmentsMat" folder
            bounding_box_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bb")
            background_segmentation_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bs")

            # Retrieve the bounding box matrices from the dataset
            bounding_box_mat = mat73.loadmat(os.path.join(bounding_box_folder, video+".mat"))
            bounding_boxes_list = bounding_box_mat['Masks']

            # Retrieve the background mask matrices from the dataset
            background_segmentation_mat = mat73.loadmat(os.path.join(background_segmentation_folder, video+".mat"))
            background_segmentations = background_segmentation_mat['Masks']

            # Create the folder where we'll save our frames
            with_background_folder_path = os.path.join(args.root_dir, subject, "withBackground", video)
            without_background_folder_path = os.path.join(args.root_dir, subject, "withoutBackground", video)
            print(f'Making folder {with_background_folder_path}')
            os.makedirs(with_background_folder_path, exist_ok=True)
            os.makedirs(without_background_folder_path, exist_ok=True)

            video_frames_folder = os.path.join(videos_folder, video)
            frame_filenames = next(os.walk(video_frames_folder))[2]
            frame_filenames.sort()
            for i, frame_filename in enumerate(frame_filenames):
                # Get the frame
                frame_filepath = os.path.join(video_frames_folder, frame_filename)
                frame_data = skimage.io.imread(frame_filepath)
                frame_data = skimage.img_as_float(frame_data).astype(np.float32)

                # Retrieve the corresponding bounding box and turn it to a square
                bounding_box = bounding_boxes_list[i]
                top_left, bottom_right = bb_box_array_to_coords(bounding_box)
                square_top_left, square_bottom_right = bb_box_to_square_bb(top_left, bottom_right)

                # Remove the background
                frame_without_background = frame_data * background_segmentations[i][:,:, np.newaxis]

                # Crop the frame with the background removed and the frame with the background
                # according to the square bounding box
                cropped_without_background = frame_without_background[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1]]
                cropped_with_background = frame_data[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1]]

                # Downscale to 128x128
                small_without_background = resize(cropped_without_background, (128, 128))
                small_with_background = resize(cropped_with_background, (128, 128))

                # Save both frames to their respective folders IE: S1/withBackground/video.id/frame.png
                skimage.io.imsave(os.path.join(with_background_folder_path, frame_filename), small_with_background)
                skimage.io.imsave(os.path.join(without_background_folder_path, frame_filename), small_without_background)
