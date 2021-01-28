from pathlib import Path
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Cityscapes(Dataset):
    def __init__(self, data_root, split='train', frame_sampling_rate=2, video_length=8, hflip=False):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.video_length = video_length
        self.hflip = hflip
        self.split_videos()

    def get_meta(self, path):
        city_name, seq_num, frame_num, _ = path.name.replace('color_mask_', '').split('_')
        return city_name, seq_num, frame_num

    def split_videos(self):
        split = 'train' if self.split == 'train' else 'test'
        paths = sorted(self.data_root.glob('**/%s_A/**/color_mask_*.png' % split))

        video_paths = []
        for i, path in enumerate(paths):
            if i == 0:
                video_paths.append([path])
                continue

            latest_path = video_paths[-1][-1]
            latest_city_name, latest_n_video, latest_n_frame = self.get_meta(latest_path)
            current_city_name, current_n_video, current_n_frame = self.get_meta(path)

            latest_n_video = int(latest_n_video)
            latest_n_frame = int(latest_n_frame)

            current_n_video = int(current_n_video)
            current_n_frame = int(current_n_frame)

            if latest_city_name == current_city_name and latest_path.parent == path.parent and \
                    latest_n_video == current_n_video and latest_n_frame + 1 == current_n_frame:
                video_paths[-1].append(path)
            else:
                video_paths.append([path])

        self.video_paths = []
        num_skip_frames_to_end = self.frame_sampling_rate * (self.video_length - 1)
        for video in video_paths:
            for i in range(len(video) - num_skip_frames_to_end):
                self.video_paths.append(video[i:i + num_skip_frames_to_end + 1:self.frame_sampling_rate])

    def __getitem__(self, idx):
        hflip = transforms.functional.hflip 
        isflip = random.randint(a=0, b=1) == 1
        transform = transforms.Compose([
            hflip if self.hflip and isflip else lambda x: x,
            transforms.Resize((128, 256), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255).long())
            ])

        mask_video = []
        video_path = self.video_paths[idx]
        for path_idx, path in enumerate(video_path):
            mask = Image.open(path)
            mask = transform(mask)
            mask_video.append(mask)
            video_path[path_idx] = str(path)

        mask_video = torch.stack(mask_video)
        return mask_video, video_path 

    def __len__(self):
        return len(self.video_paths)


class CityscapesTest(Dataset):
    def __init__(self, data_root, split='test', frame_sampling_rate=1, video_length=30, hflip=False):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = 1 
        self.video_length = 30 
        self.hflip = hflip
        self.split_videos()

    def get_meta(self, path):
        city_name, seq_num, frame_num, _ = path.name.replace('color_mask_', '').split('_')
        return city_name, seq_num, frame_num

    def split_videos(self):
        split = 'train' if self.split == 'train' else 'test'
        paths = sorted(self.data_root.glob('**/%s_A/**/color_mask_*.png' % split))

        video_paths = []
        for i, path in enumerate(paths):

            if i == 0:
                video_paths.append([path])
                continue

            latest_path = video_paths[-1][-1]
            latest_city_name, latest_n_video, latest_n_frame = self.get_meta(latest_path)
            current_city_name, current_n_video, current_n_frame = self.get_meta(path)

            latest_n_video = int(latest_n_video)
            latest_n_frame = int(latest_n_frame)

            current_n_video = int(current_n_video)
            current_n_frame = int(current_n_frame)

            if latest_city_name == current_city_name and latest_path.parent == path.parent and \
                    latest_n_video == current_n_video and latest_n_frame + 1 == current_n_frame:
                video_paths[-1].append(path)
            else:
                video_paths.append([path])

        frame_sampling_rate = self.frame_sampling_rate
        clip_size = self.video_length
        skip_size = 1
        out = False
        while not out:
            clips = []
            for video_path in video_paths:
                video_len = len(video_path)
                for i in range(0, video_len, skip_size):
                    clip = []
                    indices = list(range(i, min(i+clip_size*(frame_sampling_rate), video_len), frame_sampling_rate))
                    if len(indices) < clip_size:
                        continue
                    for j in indices:
                        clip.append(video_path[j])
                    clips += [[clip[i] for i in range(1, len(clip), 3)]]
            if len(clips) > 0:
                out = True
            else:
                clip_size -= 1
                if clip_size < 0:
                    out = True
        if len(clips) == 0:
            raise
        self.clips= clips

    def __getitem__(self, idx):
        hflip = transforms.functional.hflip 
        isflip = random.randint(a=0, b=1) == 1
        transform = transforms.Compose([
            hflip if self.hflip and isflip else lambda x: x,
            transforms.Resize((128, 256), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255).long())
            ])

        mask_video = []
        edge_video = []
        video_path = self.clips[idx]
        for path_idx, path in enumerate(video_path):
            mask = Image.open(path)
            mask = transform(mask)

            mask_video.append(mask)
            video_path[path_idx] = str(path)

        mask_video = torch.stack(mask_video)
        return mask_video, video_path 

    def __len__(self):
        return len(self.clips)
