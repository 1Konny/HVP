import sys
import imageio
import numpy as np
from scipy.misc import imresize, imsave
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

exclude_recordings = ['2011_09_28_drive_%04d_sync' % i for i in range(54, 221)] + ['2011_09_29_drive_0108']
test_recordings = ['2011_09_26_drive_0104_sync', '2011_09_26_drive_0079_sync', '2011_09_26_drive_0070_sync']

preprocess_size = (128, 160)

label_interpolation = 'nearest'
image_interpolation = 'bilinear'
semantic_label_type = 'color_mask'

root = Path('datasets_raw/KITTI/')
newroot_svg = Path('structure_generator/datasets/KITTI_64')
newroot_v2v = Path('image_generator/datasets/KITTI_vid2vid_90')

img_paths = list(root.glob('images/**/image_03/data/*.png'))
mask_paths = list(root.glob('semantic_labels/**/image_03/data/*.png'))


def process_im(im, desired_sz, interp):
    assert interp in ['bilinear', 'nearest']
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), interp=interp)
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    im_64 = imresize(im, (64, 64), interp)
    im_90 = imresize(im, (90, 90), interp)
    return im_64, im_90


def sanity_check(mode, path):
    if mode in ['P', 'L']:
        assert semantic_label_type in str(path)
    elif mode == 'RGB':
        assert path.name.startswith('00')
    else:
        raise TypeError('Invalid Image mode: %s' % mode)


def preprocessor(path):
    img = Image.open(path)
    mode = img.mode
    sanity_check(mode, path)

    recording = path.parts[-4]
    if recording in exclude_recordings:
        return
    elif recording in test_recordings:
        split = 'test'
    else:
        split = 'train'
        

    if mode == 'RGB':
        interpolation = image_interpolation
        newpath_svg = newroot_svg.joinpath(*path.parts[path.parts.index('images')+1:])
        newpath_v2v = newroot_v2v.joinpath(split + '_B', recording, path.name.replace('color_mask_', ''))
    elif mode in ['P', 'L']:
        interpolation = label_interpolation
        newpath_svg = newroot_svg.joinpath(*path.parts[path.parts.index('semantic_labels')+1:])
        newpath_v2v = newroot_v2v.joinpath(split + '_A', recording, path.name.replace('color_mask_', ''))
    else:
        raise

    newpath_svg.parent.mkdir(parents=True, exist_ok=True)
    newpath_v2v.parent.mkdir(parents=True, exist_ok=True)

    img_64, img_90 = process_im(np.asarray(Image.open(path)), preprocess_size, interpolation)
    imageio.imwrite(newpath_svg, img_64)
    imageio.imwrite(newpath_v2v, img_90)
    return

from multiprocessing import Pool
with Pool(processes=32) as pool:
    pool.map(preprocessor, img_paths)
    pool.map(preprocessor, mask_paths)
