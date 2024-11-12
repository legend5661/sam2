import os
import numpy as np
import torch
from torch.utils.data import Dataset

from func_3d.utils import random_click_3d, generate_bbox_3d

class MyDataSet_3D(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
        # Set the data list for training
        self.name_list = [f for f in os.listdir(os.path.join(data_path, mode, 'image')) if f.endswith('.npy')]
        
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name.replace('.npy', '_label.npy'))
        
        # Load 3D image and mask patches
        img_3d = np.load(img_path)
        mask_3d = np.load(mask_path)
        
        num_frame = img_3d.shape[-1]
        
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        
        img_tensor = torch.zeros(video_length, img_3d.shape[0], self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img = img_3d[..., frame_index]
            mask = mask_3d[..., frame_index]
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = torch.tensor(obj_mask).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click_3d(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox_3d(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
            
            img = torch.tensor(img).permute(2, 0, 1)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }