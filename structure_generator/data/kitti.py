import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

val_recordings = ['2011_09_26_drive_0005_sync']
test_recordings = ['2011_09_26_drive_0104_sync', '2011_09_26_drive_0079_sync', '2011_09_26_drive_0070_sync']
exclude_recordings = ['2011_09_28_drive_%04d_sync' % i for i in range(54, 221)] + ['2011_09_29_drive_0108_sync']


class KITTI(Dataset):
    def __init__(self, data_root, split='train', frame_sampling_rate=2, video_length=8, hflip=False, **kwargs):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.video_length = video_length
        self.hflip = hflip
        self.split_videos()

    def get_metadata(self, path):
        datetime = path.parent.parent.parent.name.split('_')
        datetime = '_'.join(datetime[:-3] + [datetime[-2]])
        frame_num = int(path.stem.split('_')[-1])
        return datetime, frame_num

    def split_videos(self):
        paths = list(self.data_root.glob('**/color_mask_*.png'))

        for recording in exclude_recordings:
            paths = list(filter(lambda path: recording not in str(path), paths)) 

        if self.split == 'train':
            for recording in val_recordings:
                paths = list(filter(lambda path: recording not in str(path), paths)) 
            for recording in test_recordings:
                paths = list(filter(lambda path: recording not in str(path), paths)) 
        elif self.split == 'val':
            for recording in val_recordings:
                paths = list(filter(lambda path: recording in str(path), paths)) 
        elif self.split == 'test':
            paths_ = []
            for recording in test_recordings:
                paths_ += list(filter(lambda path: recording in str(path), paths)) 
            paths = paths_
        paths = sorted(paths, key=lambda path: (path.parent.parent.parent.name, int(path.stem.split('_')[-1])))

        video_paths = []
        for i, path in enumerate(paths):
            if i == 0:
                video_paths.append([path])
                continue

            latest_path = video_paths[-1][-1]
            latest_n_video, latest_n_frame = self.get_metadata(latest_path)
            current_n_video, current_n_frame = self.get_metadata(path)

            latest_n_video = int(latest_n_video)
            latest_n_frame = int(latest_n_frame)

            current_n_video = int(current_n_video)
            current_n_frame = int(current_n_frame)

            if latest_path.parent == path.parent and \
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
        transform = transforms.Compose([
            hflip if self.hflip and random.randint(a=0, b=1) == 1 else lambda x: x,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255).long())
            ])

        video = []
        video_path = self.video_paths[idx]
        for path_idx, path in enumerate(video_path):
            mask = Image.open(path)
            mask = transform(mask)
            video.append(mask)
            video_path[path_idx] = str(path)

        video = torch.stack(video)
        return video, video_path 

    def __len__(self):
        return len(self.video_paths)


class KITTITest(Dataset):
    def __init__(self, data_root, split='test', frame_sampling_rate=1, video_length=30, hflip=False, **kwargs):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.video_length = video_length
        self.hflip = hflip
        self.split_videos()

    def split_videos(self):
        paths = list(self.data_root.glob('**/color_mask_*.png'))

        for recording in exclude_recordings:
            paths = list(filter(lambda path: recording not in str(path), paths)) 

        if self.split == 'train':
            for recording in val_recordings:
                paths = list(filter(lambda path: recording not in str(path), paths)) 
            for recording in test_recordings:
                paths = list(filter(lambda path: recording not in str(path), paths)) 
        elif self.split == 'val':
            for recording in val_recordings:
                paths = list(filter(lambda path: recording in str(path), paths)) 
        elif self.split == 'test':
            paths_ = []
            for recording in test_recordings:
                paths_ += list(filter(lambda path: recording in str(path), paths)) 
            paths = paths_

        paths = sorted(paths)
        recordings = sorted(list(set([list(path.parents)[2].name for path in paths])))
        video_paths = [sorted(list(filter(lambda path: recording in str(path), paths))) for recording in recordings]

        frame_sampling_rate = self.frame_sampling_rate
        clip_size = self.video_length
        skip_size = 5
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
                    clips.append(clip)
            if len(clips) >= 120:
                out = True
            else:
                clip_size -= 1
                if clip_size < 0:
                    out = True
        if len(clips) == 0:
            raise
        self.clips = clips

    def __getitem__(self, idx):
        hflip = transforms.functional.hflip 
        transform = transforms.Compose([
           hflip if self.hflip and random.randint(a=0, b=1) == 1 else lambda x: x,
           transforms.ToTensor(),
           transforms.Lambda(lambda x: x.mul(255).long())
           ])

        video = []
        video_path = self.clips[idx]
        for path_idx, path in enumerate(video_path):
            mask = Image.open(path)
            mask = transform(mask)
            video.append(mask)
            video_path[path_idx] = str(path)

        video = torch.stack(video)
        return video, video_path

    def __len__(self):
        return len(self.clips)
