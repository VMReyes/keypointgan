from posixpath import split
from data.base_dataset import BaseDataset, get_transform
from data import find_dataset_using_name
from image_folder import ImageFolder
import random
from PIL import Image

class AVADataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sample_window', type=int, default=[5, 30], nargs=2, help='')
        parser.add_argument('--no_mask', action='store_true', help='Apply segmentation mask to human36m images.')
        parser.add_argument('--skeleton_subset_size', type=int, default=0, help='')
        parser.add_argument('--skeleton_subset_seed', type=int, default=None, help='')
        parser.add_argument('--crop_to_bounding_box', action='store_true',
            help='Crop the pose frame to a square centered around the bounding box')
        return parser
        
    def initialize(self, opt):
        self.opt = opt
        self.load_images = True
        self.sample_window = opt.sample_window
        human36m_dataset = find_dataset_using_name("simplehuman36m")
        human36m_dataset_instance = human36m_dataset()
        human36m_dataset_instance.initialize(opt)
        self.human36m_dataset = human36m_dataset_instance
        print("dataset [simplehuman36m] was created")

        self.frame_start_time_sec = 902
        self.ava_dataset = ImageFolder("datasets/ava/video_cut_to_bb/", return_paths = True)
        self.A_transforms = get_transform(opt)


    def __getitem__(self, index):
        # Sample a frame from AVA
        ava_frame_index = index % self.ava_dataset.__len__()
        ava_frame1, ava_frame_path1 = self.ava_dataset.__getitem__(ava_frame_index)

        # Get a sample from human36m
        simplehuman36_data = self.human36m_dataset.__getitem__(index)

        # Sample the conditional frame from AVA
        offset = self.opt.sample_window[0]
        if self.opt.phase == "train":
            window_range = self.opt.sample_window[1]
            random_offset = random.randint(0, window_range)
            offset += random_offset
        
        offset_index = (ava_frame_index + offset) % self.ava_dataset.__len__()
        ava_frame2, ava_frame_path2 = self.ava_dataset.__getitem__(offset_index)

        # Perform the transforms we want on both AVA frames
        transformed_ava_frame1 = self.A_transforms(ava_frame1)
        transformed_ava_frame2 = self.A_transforms(ava_frame2)

        # Overwrite the frames from the human36 sample
        simplehuman36_data['A'] = transformed_ava_frame1
        simplehuman36_data['A_paths'] = ava_frame_path1
        simplehuman36_data['cond_A'] = transformed_ava_frame2
        simplehuman36_data['cond_A_path'] = ava_frame_path2
        return simplehuman36_data

    def __len__(self):
        return self.ava_dataset.__len__()
    def name(self):
        return "AVADataset"