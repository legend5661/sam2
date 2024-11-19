""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click_3d, generate_bbox_3d


class MyDataSet_3D(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))
        
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
        newsize = (self.img_size, self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        
        data_seg_4d_shape = np.load(mask_path + f'/{name}_seg_patch_0_label.npy').shape #(C,D,H,W)
        data_4d_shape = np.load(img_path + f'/{name}_patch_0.npy').shape
        C, D, H, W = data_4d_shape

        # num_frame = len(os.listdir(mask_path))
        import glob
        # 获取 mask_path 目录下所有 .npy 文件的数量
        num_frame = len(glob.glob(os.path.join(mask_path, '*.npy')))

        # print('num_frame_ori:',num_frame)
        data_seg_4d = np.zeros(data_seg_4d_shape + (num_frame,))
        for i in range(num_frame):
            data_seg_4d[..., i] = np.load(os.path.join(mask_path, f'{name}_seg_patch_{i}_label.npy'))

        starting_frame_nonzero = 0
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        if(data_seg_4d.shape[-1] > video_length):
            for i in range(data_seg_4d.shape[-1]):
                if np.sum(data_seg_4d[..., i]) > 0:
                    data_seg_4d = data_seg_4d[..., i:]
                    break
            starting_frame_nonzero = i
        if(data_seg_4d.shape[-1] > video_length):
            for j in reversed(range(data_seg_4d.shape[-1])):
                if np.sum(data_seg_4d[..., j]) > 0:
                    data_seg_4d = data_seg_4d[..., :j+1]
                    break
        # data_seg_4d此时形状为（C,D,H,W,num_frame）
        if(data_seg_4d.shape[-1] < video_length):
            data_seg_4d = np.concatenate((data_seg_4d, np.zeros(data_seg_4d.shape[:-1] + (video_length - data_seg_4d.shape[-1],))), axis=-1)
        num_frame = data_seg_4d.shape[-1]
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, C, D, H, W)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}
        # print('num_frame,video_length:',num_frame,video_length)
        # print('starting_frame:',starting_frame)
        # exit()
        for frame_index in range(starting_frame, starting_frame + video_length):
            img = np.load(os.path.join(img_path, f'{name}_patch_{frame_index + starting_frame_nonzero}.npy'))
            mask = data_seg_4d[..., frame_index]
            
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
                # if self.transform_msk:
                obj_mask = torch.tensor(obj_mask)
                # print(obj_mask.shape)
                obj_mask = obj_mask.squeeze(0)
                obj_mask = obj_mask.reshape(newsize)
                obj_mask = obj_mask.unsqueeze(0).int()
                    # obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click_3d(np.array(obj_mask.squeeze(0)), point_label, seed=123)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox_3d(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
            # if self.transform:
                # state = torch.get_rng_state()
                # img = self.transform(img)
                # torch.set_rng_state(state)
            new_img_size = (C,self.img_size, self.img_size, self.img_size)
            img = torch.tensor(np.array(img)).reshape(new_img_size)

            img_tensor[frame_index - starting_frame, :, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':name}
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }