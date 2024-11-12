""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg

args = cfg.parse_args()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []



def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
    hard = 0  # 用于记录困难样本的数量，未使用
    epoch_loss = 0  # 记录每个epoch的总损失
    epoch_prompt_loss = 0  # 记录每个epoch的提示损失
    epoch_non_prompt_loss = 0  # 记录每个epoch的非提示损失
    ind = 0  # 未使用的索引变量

    # 设置模型为训练模式
    net.train()
    
    # 清空优化器的梯度
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    
    video_length = args.video_length  # 从参数中获取视频长度

    # 设置 GPU 设备
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt  # 获取提示类型（如点击或边界框）
    prompt_freq = args.prompt_freq  # 获取提示频率

    # 定义损失函数
    lossfunc = criterion_G  # 使用指定的损失函数

    # 使用 tqdm 创建进度条
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()  # 清空 CUDA 缓存以释放内存
            imgs_tensor = pack['image']  # 获取图像张量
            mask_dict = pack['label']  # 获取标签字典
            
            # 根据提示类型获取相应的数据
            if prompt == 'click':
                pt_dict = pack['pt']  # 获取点击点字典
                point_labels_dict = pack['p_label']  # 获取点击点标签字典
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']  # 获取边界框字典
            
            imgs_tensor = imgs_tensor.squeeze(0)  # 去掉多余的维度
            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=GPUdevice)  # 转换数据类型并转移到GPU
            
            # 初始化训练状态
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))  # 根据提示频率生成提示帧的索引
            obj_list = []  # 存储对象列表

            # 收集提示帧中的所有对象ID
            # prompt_frame_id用于指定哪些帧需要提示
            # 这一步是将每一个需要提示的帧的所有对象ID收集起来
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))  # 去重
            if len(obj_list) == 0:  # 如果没有对象，则跳过
                continue

            name = pack['image_meta_dict']['filename_or_obj']  # 获取图像文件名
            
            # 使用自动混合精度进行前向传播
            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:  # 对于每个提示帧
                    for ann_obj_id in obj_list:  # 对于每个对象ID
                        # 对于每一个被提示的帧中的每一个对象，依次添加提示来进行调优
                        try:
                            if prompt == 'click':  # 如果提示类型是点击
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)  # 获取点击点
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)  # 获取点击点标签
                                # 将新点击点添加到训练状态中
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':  # 如果提示类型是边界框
                                bbox = bbox_dict[id][ann_obj_id]  # 获取边界框
                                # 将边界框添加到训练状态中
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:  # 如果未找到对象ID，添加一个空掩码
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # 初始化视频分割结果字典
            
                # 通过训练状态传播视频中的分割结果
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0  # 初始化总损失
                non_prompt_loss = 0  # 初始化非提示损失
                prompt_loss = 0  # 初始化提示损失
                
                # 对于每一帧和每个对象计算损失
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]  # 获取预测结果
                        pred = pred.unsqueeze(0)  # 增加一个维度
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)  # 获取真实标签
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)  # 如果未找到标签，则使用全零掩码
                        
                        # 可视化训练过程
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))  # 显示输入图像
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')  # 显示预测掩码
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]  # 获取边界框
                                # 在预测图上绘制边界框
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')  # 显示真实标签
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)  # 保存可视化结果
                            plt.close()
                        
                        obj_loss = lossfunc(pred, mask)  # 计算损失
                        loss += obj_loss.item()  # 累加损失
                        if id in prompt_frame_id:  # 如果是提示帧，计算提示损失
                            prompt_loss += obj_loss
                        else:  # 否则，计算非提示损失
                            non_prompt_loss += obj_loss
                
                # 计算平均损失
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})  # 更新进度条显示
                epoch_loss += loss  # 累加总损失
                epoch_prompt_loss += prompt_loss.item()  # 累加提示损失
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()  # 累加非提示损失

                # 梯度反向传播
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)  # 计算非提示损失的梯度
                    optimizer2.step()  # 更新优化器2的参数
                if optimizer1 is not None:
                    prompt_loss.backward()  # 计算提示损失的梯度
                    optimizer1.step()  # 更新优化器1的参数
                
                # 清空梯度
                optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                net.reset_state(train_state)  # 重置模型状态以便下一个batch

            pbar.update()  # 更新进度条

    # 返回每个epoch的平均损失
    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.vis:
                            os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])
