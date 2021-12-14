from dominate.tags import video
from data import find_dataset_using_name
from data.base_dataset import BaseDataset, get_transform
from image_folder import ImageFolder
import torchvision.transforms as transforms

def get_transform(opt, channels=3):
    mean = 0.5
    std = 0.5
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize([mean] * channels,
                                           [std] * channels)]
    return transforms.Compose(transform_list)

class MoviesDatasetSingle(object):
    def __init__(self, root):
        self.root = root
        pass
    def get_pair(self, frame1, frame2):
        pass
    def get_item(self, index):
        pass
    def num_samples(self):
        pass

class MoviesAndHumansDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sample_window', type=int, default=[5, 30], nargs=2, help='')
        parser.add_argument('--no_mask', action='store_true', help='Apply segmentation mask to human36m images.')
        parser.add_argument('--video_dir', help='Path to folder with video frames.')
        parser.add_argument('--pose_source', required = True, help='The source of real_A (pose) images.')
        parser.add_argument('--appearance_source', required = True, help='The source of cond_A (appearance) images.')
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

        self.video_dataset = ImageFolder(opt.video_dir, return_paths=True)
        self.A_transforms = get_transform(opt)

    def __getitem__(self, index):
        video_frame, video_frame_path = self.video_dataset.__getitem__(index)
        transformed_video_frame = self.A_transforms(video_frame)
        simplehuman36_data = self.human36m_dataset.__getitem__(index)
        if self.opt.appearance_source == "video" and self.opt.pose_source == "simplehuman36m":
            simplehuman36_data['cond_A'] = transformed_video_frame 
            simplehuman36_data['cond_A_path'] = video_frame_path
        elif self.opt.appearance_source == "simplehuman36m" and self.opt.pose_source == "video":
            simplehuman36_data['A'] = transformed_video_frame
            simplehuman36_data['A_paths'] = video_frame_path 
        elif self.opt.appearance_source == "video" and self.opt.pose_source == "video":
            simplehuman36_data['A'] = transformed_video_frame
            simplehuman36_data['A_paths'] = video_frame_path 
            try:
                # get second frame and put it in
                offset = self.opt.sample_window[0]
                second_video_frame, second_video_frame_path = self.video_dataset.__getitem__(index+offset)
                second_transformed_video_frame = self.A_transforms(second_video_frame)
                simplehuman36_data['cond_A'] = second_transformed_video_frame
                simplehuman36_data['cond_A_path'] = second_video_frame_path
            except:
                # repeat the frame we got above
                simplehuman36_data['cond_A'] = transformed_video_frame 
                simplehuman36_data['cond_A_path'] = video_frame_path 
        else:
            assert False
        return simplehuman36_data
    def __len__(self):
        return min(self.human36m_dataset.__len__(), self.video_dataset.__len__())
    def name(self):
        return 'MoviesAndHumansDataset'