import os
from multiprocessing import Process

def sample_frames(video_folder, video_filename):
    video_filepath = video_folder + "/" + video_filename
    save_path = video_folder + "/frames"

    try:
        os.makedirs(save_path)
    except:
        pass
    print(video_filepath)
    print('ffmpeg -i %s -ss 902 -t 986 -r 1 "%s/frame-%%06d.jpg"' % (video_filepath, save_path))
    os.system('ffmpeg -i %s -ss 902 -t 986 -r 1 "%s/frame-%%06d.jpg"' % (video_filepath, save_path))
    os.system('rm %s' % (video_filepath))

video_filenames_file = open("util/video_list_train.txt")
video_filenames = video_filenames_file.readlines()

processes = []
for video_filename in video_filenames:
    # download video
    video_filename_clean = video_filename.strip()
    video_id = video_filename_clean.split('.')[0]
    try:
        os.makedirs("datasets/ava/videos")
    except:
        pass
    save_dir = "datasets/ava/videos"
    os.system('wget -P %s/%s https://s3.amazonaws.com/ava-dataset/trainval/%s' % (save_dir, video_id, video_filename_clean))
    p = Process(target=sample_frames, args=(save_dir + "/" + video_id, video_filename_clean))
    p.start()
    processes.append(p)

for p in processes:
    p.join()