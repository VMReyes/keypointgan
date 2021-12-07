import os, sys
import mat73
import argparse
import skimage.io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

parser = argparse.ArgumentParser(description='Simplifies a Human3.6m dataset after running subject_videos_to_frames.sh')
parser.add_argument('--root_dir', required=True,
                    help='Root directory of the Human3.6m dataset after converting videos to frames using subject_videos_to_frames.sh')

SQUARE_IMG_SIZE = 128

def adjust_landmarks_to_bounding_box(raw_landmarks, top_left, bottom_right):
    """Returns normalized landmarks by translating input landmarks (array of 64, 1)
    with respect to the square bounding box, and then scaling them to SQUARE_IMG_SIZE 
    """
    landmarks = []
    bb_height = bottom_right[0] - top_left[0]
    bb_width = bottom_right[1] - top_left[1]
    for i in range(0, len(raw_landmarks), 2):
        x = raw_landmarks[i]
        y = raw_landmarks[i+1]
        translated_x = (int) (((float) (x - top_left[1]) / bb_width) * SQUARE_IMG_SIZE)
        translated_y = (int) (((float) (y - top_left[0]) / bb_height) * SQUARE_IMG_SIZE)
        assert translated_x >= 0 and translated_x < 128
        assert translated_y >= 0 and translated_y < 128
        landmarks.append([translated_x, translated_y])
    return landmarks

def get_bounding_boxes(subject, video):
    bounding_box_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bb")
    # Retrieve the bounding box matrices from the dataset
    bounding_box_mat = mat73.loadmat(os.path.join(bounding_box_folder, video+".mat"))
    bounding_boxes_list = bounding_box_mat['Masks']
    return bounding_boxes_list

def get_landmarks(subject, video):
    raw_landmarks_folder = os.path.join(args.root_dir, subject, "Landmarks_raw")
    # Load relevant landmarks
    raw_landmarks_filepath = os.path.join(raw_landmarks_folder, video + ".mat")
    raw_landmarks_data = scipy.io.loadmat(raw_landmarks_filepath)['keypoints_2d']
    return raw_landmarks_data

def get_background_mask(subject, video):
    background_segmentation_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bs")
    background_segmentation_mat = mat73.loadmat(os.path.join(background_segmentation_folder, video+".mat"))
    background_segmentations = background_segmentation_mat['Masks']
    return background_segmentations

def create_simplified_humans36m_dataset(args):
    subject_folders = next(os.walk(args.root_dir))[1]
    for subject in subject_folders:
        print(f'Processing subject: {subject}')
        # Expecting a "Videos" folder in every actor's folder
        videos_folder = os.path.join(args.root_dir, subject, "Videos")
        # Expecting a folder for every video from that subject,
        # where the video has been converted to its frames
        frame_by_frame_video_folders = next(os.walk(videos_folder))[1]
        frame_by_frame_video_folders.sort()
        for video in frame_by_frame_video_folders:
            print(f'Processing the landmarks for video: {video}')
            bounding_boxes_list = get_bounding_boxes(subject, video)
            raw_landmarks_data = get_landmarks(subject, video)
            background_segmentations = get_background_mask(subject, video)
            # Create the folder where we'll save our frames
            with_background_folder_path = os.path.join(args.root_dir, subject, "withBackground", video)
            without_background_folder_path = os.path.join(args.root_dir, subject, "withoutBackground", video)
            print(f'Making folder {with_background_folder_path}')
            os.makedirs(with_background_folder_path, exist_ok=True)
            os.makedirs(without_background_folder_path, exist_ok=True)

            # Create the folder where we'll save our new landmarks
            translated_landmarks_folder_path = os.path.join(args.root_dir, subject, "Landmarks")
            print(f'Making folder {translated_landmarks_folder_path}')
            os.makedirs(translated_landmarks_folder_path, exist_ok=True)

            video_frames_folder = os.path.join(videos_folder, video)
            frame_filenames = next(os.walk(video_frames_folder))[2]
            frame_filenames.sort()
            print(f'Have {len(bounding_boxes_list)} Bounding boxes and {len(background_segmentations)} Background Segmentations for {len(frame_filenames)} Frames.')
            print(f'Resizing frames to {min(len(bounding_boxes_list), len(background_segmentations))}.')

            frame_filenames = frame_filenames[:min(len(bounding_boxes_list), len(background_segmentations))]
            print(f'New amount of frames: {len(frame_filenames)}.')
            landmarks = []
            for i, frame_filename in enumerate(frame_filenames):
                print(f'Processing frame {i}; filename: {frame_filename}')
                # Get the frame
                frame_filepath = os.path.join(video_frames_folder, frame_filename)
                frame_data = skimage.io.imread(frame_filepath)
                frame_data = skimage.img_as_float(frame_data).astype(np.float32)

                # Retrieve the corresponding bounding box and turn it to a square
                bounding_box = bounding_boxes_list[i]
                top_left, bottom_right = bb_box_array_to_coords(bounding_box)
                square_top_left, square_bottom_right = bb_box_to_square_bb(top_left, bottom_right, frame_data.shape[1], frame_data.shape[0])

                translated_landmarks = adjust_landmarks_to_bounding_box(raw_landmarks_data, square_top_left, square_bottom_right)
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
                save_path_with_background = os.path.join(with_background_folder_path, frame_filename)
                save_path_without_background = os.path.join(without_background_folder_path, frame_filename)
                skimage.io.imsave(save_path_with_background, small_with_background)
                skimage.io.imsave(save_path_without_background, small_without_background)

                landmarks.append(translated_landmarks)
            scipy.io.savemat(os.path.join(translated_landmarks_folder_path, video + ".mat"), {"keypoints_2d" : landmarks})
            

def adjust_landmarks_files_to_crop():
    subject_folders = next(os.walk(args.root_dir))[1]
    for subject in subject_folders:
        print(f'Processing the landmarks for subject: {subject}')
        # Expecting a "Videos" folder in every actor's folder
        videos_folder = os.path.join(args.root_dir, subject, "Videos")
        # Expecting a folder for every video from that subject,
        # where the video has been converted to its frames
        frame_by_frame_video_folders = next(os.walk(videos_folder))[1]
        frame_by_frame_video_folders.sort()
        frame_by_frame_video_folders.reverse()
        for video in frame_by_frame_video_folders:
            print(f'Processing the landmarks for video: {video}')
            # Expecting a "ground_truth_bb" folder in every "SegmentsMat" folder
            # Expecting a "Landmarks_raw" folder in every actor's folder
            bounding_box_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bb")
            raw_landmarks_folder = os.path.join(args.root_dir, subject, "Landmarks_raw")

            # Load relevant landmarks
            raw_landmarks_filepath = os.path.join(raw_landmarks_folder, video + ".mat")
            raw_landmarks_data = scipy.io.loadmat(raw_landmarks_filepath)['keypoints_2d']

            # Retrieve the bounding box matrices from the dataset
            bounding_box_mat = mat73.loadmat(os.path.join(bounding_box_folder, video+".mat"))
            bounding_boxes_list = bounding_box_mat['Masks']

            # Iterate over the frames in the video
            video_frames_folder = os.path.join(videos_folder, video)
            frame_filenames = next(os.walk(video_frames_folder))[2]
            frame_filenames.sort()
            print(f'Have {len(bounding_boxes_list)} Bounding boxes  for {len(frame_filenames)} Frames.')
            print(f'Resizing frames to {len(bounding_boxes_list)}.')

            frame_filenames = frame_filenames[:len(bounding_boxes_list)]
            print(f'New amount of frames: {len(frame_filenames)}.')

            # Iterate over the frames in the video
            translated_landmarks = []
            for i, frame_filename in enumerate(frame_filenames):
                print(f'Processing frame {i}; filename: {frame_filename}')
                frame_filepath = os.path.join(video_frames_folder, frame_filename)
                frame_data = skimage.io.imread(frame_filepath)
                frame_data = skimage.img_as_float(frame_data).astype(np.float32)

                # Retrieve the corresponding bounding box and turn it to a square
                bounding_box = bounding_boxes_list[i]
                top_left, bottom_right = bb_box_array_to_coords(bounding_box)

                square_top_left, square_bottom_right = bb_box_to_square_bb(top_left, bottom_right, frame_data.shape[1], frame_data.shape[0])
                relevant_landmarks = raw_landmarks_data[i]
                processed_landmarks = adjust_landmarks_to_bounding_box(relevant_landmarks, square_top_left, square_bottom_right)
                translated_landmarks.append(processed_landmarks)
            translated_landmarks_folder_path = os.path.join(args.root_dir, subject, "Landmarks")
            scipy.io.savemat(os.path.join(translated_landmarks_folder_path, video + ".mat"), {"keypoints_2d" : translated_landmarks})

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
    bot_left = None
    for x in range(top_left[0], len(bbox)):
        y = top_left[1]
        if bbox[x, y] == False:
            break
        bot_left = (x, y)

    bottom_right = None
    for y in range(bot_left[1], len(bbox[0])):
        x = bot_left[0]
        if bbox[x, y] == False:
            break
        bottom_right = (x, y)
    return top_left, bottom_right

def bb_box_to_square_bb(top_left, bottom_right, im_width, im_height):
    # Calculate height and width of bounding box
    height = bottom_right[0] - top_left[0]
    width = bottom_right[1] - top_left[1]
    if height > width:
        height_offset = int(height * 0.125)
        new_top = max(0, top_left[0] - height_offset)
        new_bottom = min(im_height - 1, bottom_right[0] + height_offset)
        new_bb_height = height * 1.25
        offset = int((new_bb_height - width) / 2)
        new_left = max(0, top_left[1] - offset)
        new_right = min(im_width - 1, bottom_right[1]+offset)
        new_top_left = (new_top, new_left)
        new_bottom_right = (new_bottom, new_right)
        return new_top_left, new_bottom_right
    else:
        width_offset = int(width / 2.0)
        new_left = max(0, top_left[1] - width_offset)
        new_right = max(im_width - 1, bottom_right[1] + width_offset)
        new_top_left = (top_left[0], new_left)
        new_bottom_right = (bottom_right[0], new_right)
        return new_top_left, new_bottom_right

if __name__ == '__main__':
    args = parser.parse_args()
    adjust_landmarks_files_to_crop()
    exit()
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
            # Expecting a "Landmarks_raw" folder in every actor's folder
            bounding_box_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bb")
            background_segmentation_folder = os.path.join(args.root_dir, subject, "MySegmentsMat", "ground_truth_bs")
            landmarks_raw_folder = os.path.join(args.root_dir, subject, "Landmarks_raw")

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

            # Create the folder where we'll save our new landmarks
            translated_landmarks_folder_path = os.path.join(args.root_dir, subject, "Landmarks", video)
            print(f'Making folder {translated_landmarks_folder_path}')
            os.makedirs(translated_landmarks_folder_path, exist_ok=True)

            video_frames_folder = os.path.join(videos_folder, video)
            frame_filenames = next(os.walk(video_frames_folder))[2]
            frame_filenames.sort()
            print(f'Have {len(bounding_boxes_list)} Bounding boxes and {len(background_segmentations)} Background Segmentations for {len(frame_filenames)} Frames.')
            print(f'Resizing frames to {min(len(bounding_boxes_list), len(background_segmentations))}.')

            frame_filenames = frame_filenames[:min(len(bounding_boxes_list), len(background_segmentations))]
            print(f'New amount of frames: {len(frame_filenames)}.')

            for i, frame_filename in enumerate(frame_filenames):
                print(f'Processing frame {i}; filename: {frame_filename}')
                save_path_with_background = os.path.join(with_background_folder_path, frame_filename)
                save_path_without_background = os.path.join(without_background_folder_path, frame_filename)
                translated_landmarks_filepath = os.path.join(translated_landmarks_folder_path, "frame-%06d.mat" % (i+1))
                if os.path.exists(save_path_without_background) and os.path.exists(save_path_with_background) and os.path.exists(translated_landmarks_filepath):
                    print(f'Skipping frame {i}; filename: {frame_filename}. Both exist already and landmarks exists.')
                    continue
                elif os.path.exists(save_path_without_background) and os.path.exists(save_path_with_background):
                    print(f'Skipping frame {i}; filename: {frame_filename}. Both exist, but landmarks dont.')
                    print("Generating the landmarks file, however.")
                    # Get the frame
                    frame_filepath = os.path.join(video_frames_folder, frame_filename)
                    frame_data = skimage.io.imread(frame_filepath)
                    frame_data = skimage.img_as_float(frame_data).astype(np.float32)
                    # Retrieve the corresponding bounding box and turn it to a square
                    bounding_box = bounding_boxes_list[i]
                    top_left, bottom_right = bb_box_array_to_coords(bounding_box)

                    square_top_left, square_bottom_right = bb_box_to_square_bb(top_left, bottom_right, frame_data.shape[1], frame_data.shape[0])
                    landmarks_filepath = os.path.join(landmarks_raw_folder, video, "frame-%06d.mat" % (i+1))
                    translated_landmarks = adjust_landmarks_to_bounding_box(landmarks_filepath, square_top_left, square_bottom_right)
                    scipy.io.savemat(translated_landmarks_filepath, {'keypoints_2d': translated_landmarks})
                    continue
                # Get the frame
                frame_filepath = os.path.join(video_frames_folder, frame_filename)
                frame_data = skimage.io.imread(frame_filepath)
                frame_data = skimage.img_as_float(frame_data).astype(np.float32)

                # Retrieve the corresponding bounding box and turn it to a square
                bounding_box = bounding_boxes_list[i]
                top_left, bottom_right = bb_box_array_to_coords(bounding_box)

                square_top_left, square_bottom_right = bb_box_to_square_bb(top_left, bottom_right, frame_data.shape[1], frame_data.shape[0])

                # Create the new landmarks TODO(vmreyes)
                landmarks_filepath = os.path.join(landmarks_raw_folder, video, "frame-%06d.mat" % (i+1))
                translated_landmarks = adjust_landmarks_to_bounding_box(landmarks_filepath, square_top_left, square_bottom_right)
                translated_landmarks_filepath = os.path.join(translated_landmarks_folder_path, "frame-%06d.mat" % (i+1))
                scipy.io.savemat(translated_landmarks_filepath, {'keypoints_2d': translated_landmarks})


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
