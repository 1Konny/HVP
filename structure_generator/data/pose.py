import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class Pose(Dataset):
    def __init__(self, data_root, split='test', frame_sampling_rate=1, video_length=30, hflip=False, **kwargs):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.video_length = video_length
        self.hflip = hflip
        self.split_videos()

    def get_metadata(self, path):
        video_idx = path.parent.parent.name.strip('V')
        clip_idx = path.parent.name.strip('C')
        video_id = '%s_%s' % (video_idx, clip_idx)
        frame_idx = int(path.stem.split('_')[0].strip('frame'))
        return video_id, frame_idx

    def split_videos(self):
        import json
        split_path = self.data_root / 'split.json'
        with open(split_path, 'r') as f:
            split_info = json.load(f)

        paths = []
        for video_idx in split_info[self.split]:
            for clip_idx in split_info[self.split][video_idx]:
                clipdir = self.data_root.joinpath( 'V%05d/C%05d' % (int(video_idx), int(clip_idx)))
                paths_ = clipdir.glob('*.png')
                paths_ = list(filter(lambda path: 'IUV' in str(path), paths_))
                paths += paths_

        paths = sorted(paths)

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
            transforms.Lambda(lambda x: x[2:].mul(255).long())
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


class PoseTest(Dataset):
    def __init__(self, data_root, split='test', frame_sampling_rate=1, video_length=30, hflip=False, **kwargs):
        self.data_root = Path(data_root)
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.video_length = video_length
        self.hflip = hflip
        self.split_videos()

    def split_videos(self):
        import json
        split_path = self.data_root / 'split.json'
        with open(split_path, 'r') as f:
            split_info = json.load(f)

        paths = []
        vid_clip_idx = []
        for video_idx in split_info[self.split]:
            for clip_idx in split_info[self.split][video_idx]:
                clipdir = self.data_root.joinpath( 'V%05d/C%05d' % (int(video_idx), int(clip_idx)))
                vid_clip_idx.append('V%05d/C%05d' % (int(video_idx), int(clip_idx)))
                paths_ = clipdir.glob('*.png')
                paths_ = list(filter(lambda path: 'IUV' in str(path), paths_))
                paths += paths_

        paths = sorted(paths)
        video_paths = [sorted(list(filter(lambda path: recording in str(path), paths))) for recording in vid_clip_idx]

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
            if len(clips) > 3:
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
            transforms.Lambda(lambda x: x[2:].mul(255).long())
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
