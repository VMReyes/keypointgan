import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms.functional as functional
import os

def tight_bounding_box_to_square(top, bottom, left, right):
    bbox_height = bottom - top
    bbox_width = right - left
    square_length = int(max(bbox_height, bbox_width) * 1.25)
    square_top = top - int((square_length - bbox_height) / 2.0)
    square_left = left - int((square_length - bbox_width) / 2.0)
    return (square_top, square_left, square_length)

def crop_and_save(video_id, frame_num, person_id, box):
    with Image.open("datasets/ava/videos/%s/frames/frame-%06d.jpg" % (video_id, frame_num - 902 + 1)) as im:
        width, height= im.size
        left, top = (box[0][0] * width, box[0][1] * height)
        right, bottom = (box[1][0] * width, box[1][1] * height)
        square_bb_top, square_bb_left, square_bb_length = tight_bounding_box_to_square(top, bottom, left, right)
        cropped_im = functional.resized_crop(im, square_bb_top, square_bb_left, square_bb_length, square_bb_length, 128)
        try:
            os.makedirs("datasets/ava/video_cut_to_bb/%s" % (video_id))
        except:
            pass
        cropped_im.save("datasets/ava/video_cut_to_bb/%s/%06d-%03d.jpg" % (video_id, frame_num - 902 + 1, person_id))


def bbox_overlap(bbox1, bbox2):
    top_left1, bottom_right1 = bbox1
    top_left2, bottom_right2 = bbox2

    if ((top_left1[0] >= bottom_right2[0]) or (top_left2[0] >= bottom_right1[0])):
        return False
    
    if ((bottom_right1[1] <= top_left2[1]) or (bottom_right2[1] <= top_left1[1])):
        return False
    return True

def draw_box(video_id, frame_num, bbox1):
    with Image.open("datasets/ava/videos/%s/frames/frame-%06d.jpg" % (video_id, frame_num - 902 + 1)) as im:
        width, height= im.size
        top_left_box1 = (bbox1[0][0] * width, bbox1[0][1] * height)
        bottom_right_box1 = (bbox1[1][0] * width, bbox1[1][1] * height)
        im_with_box = ImageDraw.Draw(im)
        im_with_box.rectangle((top_left_box1, bottom_right_box1))
        random_int = random.randint(0,10)
        im.save("debug/bounding_boxes/%s-%06d-%d.jpg" % (video_id, frame_num - 902 + 1, random_int))

def draw_boxes(video_id, frame_num, bbox1, bbox2):
    with Image.open("datasets/ava/videos/%s/frames/frame-%06d.jpg" % (video_id, frame_num - 902 + 1)) as im:
        width, height= im.size
        top_left_box1 = (bbox1[0][0] * width, bbox1[0][1] * height)
        bottom_right_box1 = (bbox1[1][0] * width, bbox1[1][1] * height)
        top_left_box2 = (bbox2[0][0] * width, bbox2[0][1] * height)
        bottom_right_box2 = (bbox2[1][0] * width, bbox2[1][1] * height)
        im_with_box = ImageDraw.Draw(im)
        im_with_box.rectangle((top_left_box1, bottom_right_box1))
        im_with_box.rectangle((top_left_box2, bottom_right_box2))

        im.save("debug/overlapped_bounding_boxes/%s-%06d-.jpg" % (video_id, frame_num - 902 + 1))

if __name__ == '__main__':
    bounding_boxes = dict()
    with open('datasets/ava/annotations/ava_train_v2.2.csv') as annotation_file:
        annotation_lines = annotation_file.readlines()
        for annotation in annotation_lines:
            split_annotation = annotation.split(',')
            youtube_id = split_annotation[0]
            timestamp = int(split_annotation[1])
            bounding_box_top_left = (float(split_annotation[2]), float(split_annotation[3]))
            bounding_box_bottom_right = (float(split_annotation[4]), float(split_annotation[5]))
            person_id = int(split_annotation[7])
            if youtube_id not in bounding_boxes:
                bounding_boxes[youtube_id] = dict()
            if timestamp not in bounding_boxes[youtube_id]:
                bounding_boxes[youtube_id][timestamp] = dict()
            bounding_boxes[youtube_id][timestamp][person_id] = (bounding_box_top_left, bounding_box_bottom_right)

    overlap_counts = dict()
    for video_id in bounding_boxes.keys():
        frame_overlaps = 0
        for frame_num in bounding_boxes[video_id].keys():
            overlap = False
            frame_bboxes = []
            for person_id in bounding_boxes[video_id][frame_num].keys():
                person_bbox = bounding_boxes[video_id][frame_num][person_id]
                # perform crop and save
                crop_and_save(video_id, frame_num, person_id, person_bbox)
                #draw_box(video_id, frame_num, person_bbox)
                for other_bbox in frame_bboxes:
                    if bbox_overlap(person_bbox, other_bbox):
                        #draw_boxes(video_id, frame_num, person_bbox, other_bbox)
                        overlap = True
                frame_bboxes.append(person_bbox)
            if overlap:
                if frame_overlaps == 0:
                    #print(video_id, frame_num - 902 + 1)
                    pass
                frame_overlaps += 1
        frame_num = len(bounding_boxes[video_id].keys())
        overlap_counts[video_id] = {"overlap_count":frame_overlaps, "overlap_percent":float(frame_overlaps) / frame_num}
    
    total_frames = 0
    total_overlapped_frames = 0
    for video_id in overlap_counts:
        total_frames += len(bounding_boxes[video_id].keys())
        total_overlapped_frames += overlap_counts[video_id]['overlap_count']
    print("Total frames: %d, Total overlapped frames: %d, Total percentage of frames overlapped: %f"
            % (total_frames, total_overlapped_frames, float(total_overlapped_frames) / total_frames))

    overlap_percents = []
    for video_id in overlap_counts:
        overlap_percent = overlap_counts[video_id]['overlap_percent']
        overlap_percents.append(overlap_percent)

    plt.hist(overlap_percents, bins=50)
    plt.savefig("frame_overlap_histogram.png")

