### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.use_real = opt.use_real_img
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        if self.use_real:
            self.B_paths = sorted(make_grouped_dataset(self.dir_B))
            check_path_valid(self.A_paths, self.B_paths)
        if self.opt.use_instance:                
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
            check_path_valid(self.A_paths, self.I_paths)

        if opt.custom_data_root is not None:
            self.A_paths2 = sorted(make_grouped_dataset(opt.custom_data_root))
            def kitti_path_match(A_paths2, dataroot):
                B_paths2 = []
                for batch_dir in A_paths2:
                    B_paths2_seq = []
                    for sample_dir in batch_dir:
                        recording, color_mask_name = sample_dir.strip('/').split('/')[-2:]
                        frame_name = color_mask_name.split('_')[-1]
                        B_path2 = os.path.join(dataroot, 'test_B', recording, frame_name)
                        if os.path.exists(B_path2):
                            B_paths2_seq.append(B_path2)
                        else:
                            B_paths2_seq.append(B_paths2_seq[-1])
                    B_paths2.append(B_paths2_seq)
                return B_paths2 
            def cityscapes_path_match(A_paths2, dataroot):
                B_paths2 = []
                for batch_dir in A_paths2:
                    B_paths2_seq = []
                    for sample_dir in batch_dir:
                        recording, color_mask_name = sample_dir.strip('/').split('/')[-2:]
                        frame_name = color_mask_name.replace('color_mask_', '') 
                        B_path2 = os.path.join(dataroot, 'test_B', recording, frame_name)
                        if os.path.exists(B_path2):
                            B_paths2_seq.append(B_path2)
                        else:
                            B_paths2_seq.append(B_paths2_seq[-1])
                    B_paths2.append(B_paths2_seq)
                return B_paths2 
            if 'kitti' in opt.dataroot.lower():
                path_match = kitti_path_match
            elif 'cityscapes' in opt.dataroot.lower():
                path_match = cityscapes_path_match 
            self.B_paths2 = path_match(self.A_paths2, opt.dataroot)
            self.A_paths = self.A_paths2
            self.B_paths = self.B_paths2

        self.init_frame_idx(self.A_paths)

    def __getitem__(self, index):
        self.A, self.B, self.I, seq_idx = self.update_frame_idx(self.A_paths, index)
        tG = self.opt.n_frames_G
              
        A_img = Image.open(self.A_paths[seq_idx][0]).convert('RGB')        
        params = get_img_params(self.opt, A_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB
        frame_range = list(range(tG)) if self.A is None else [tG-1]
           
        A_paths = []
        for i in frame_range:                                                   
            A_path = self.A_paths[seq_idx][self.frame_idx + i]            
            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)            
            self.A = concat_frame(self.A, Ai, tG)

            if self.use_real:
                B_path = self.B_paths[seq_idx][self.frame_idx + i]
                Bi = self.get_image(B_path, transform_scaleB)                
                self.B = concat_frame(self.B, Bi, tG)
            else:
                self.B = 0

            if self.opt.use_instance:
                I_path = self.I_paths[seq_idx][self.frame_idx + i]
                Ii = self.get_image(I_path, transform_scaleA) * 255.0                
                self.I = concat_frame(self.I, Ii, tG)
            else:
                self.I = 0

            A_paths.append(A_path)

        self.frame_idx += 1        
        return_list = {'A': self.A, 'B': self.B, 'inst': self.I, 'A_path': A_path, 'change_seq': self.change_seq, 'A_paths':A_paths}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):        
        return sum(self.frames_count)

    def n_of_seqs(self):        
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'
