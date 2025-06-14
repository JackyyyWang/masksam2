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
import numpy as np

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
scaler = torch.amp.GradScaler("cuda")
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def region_wise_bce_loss(pred, mask, loss_func=criterion_G, bbox=None, pos_weight=None):
    """
    Compute BCE loss only within the specified bounding box region.

    Args:
        pred: Prediction tensor
        mask: Ground truth mask tensor
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max) or None
        pos_weight: Positive weight for BCE loss

    Returns:
        loss: Region-wise BCE loss
    """
    device = pred.device

    if bbox is None:
        # Fallback to regular BCE loss if no bbox is provided
        return F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Create a region mask that is 1 inside the bounding box and 0 outside
    region_mask = torch.zeros_like(pred)
    region_mask[:, :, y_min:y_max + 1, x_min:x_max + 1] = 1.0

    # Compute BCE loss element-wise
    element_wise_loss = F.binary_cross_entropy_with_logits(
        pred, mask,
        pos_weight=pos_weight,
        reduction='none'
    )

    # Apply the region mask and compute the mean over the region
    masked_loss = (element_wise_loss * region_mask).sum()
    num_pixels = region_mask.sum()

    if num_pixels > 0:
        masked_loss = masked_loss / num_pixels

    return masked_loss

def train_sam_with_mask_prompt(args, net: nn.Module, optimizer, train_loader, epoch):
    """
    Train the SAM model using mask prompts from a single slice with bidirectional propagation
    """
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0

    # Training mode
    net.train()
    if optimizer is not None:
        optimizer.zero_grad()


    video_length = args.video_length
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G
    # lossfunc = paper_loss

    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_idx, pack in enumerate(train_loader):
            torch.cuda.empty_cache()

            # Extract data from batch
            imgs_tensor = pack['image']
            mask_dict = pack['label'][0]
            prompt_slice_idx = pack['prompt_slice_idx'][0].item()

            gland_bboxes = pack['gland_bboxes'] if 'gland_bboxes' in pack else None

            # print(imgs_tensor.shape)
            # print(mask_dict)
            # print(prompt_slice_idx)
            # print(gland_bboxes)

            # Get prompt masks
            prompt_masks = {}
            if 'prompt_masks' in pack:
                prompt_masks = pack['prompt_masks']
            else:
                if prompt_slice_idx in mask_dict:
                    prompt_masks[prompt_slice_idx] = mask_dict[prompt_slice_idx]

            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=GPUdevice)

            # Initialize inference state
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)

            # Get list of unique objects across all frames
            obj_list = []
            for id in mask_dict.keys():
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))

            if len(obj_list) == 0:
                pbar.update()
                continue

            with torch.cuda.amp.autocast():
                # Add mask prompt for each object at the prompt slice
                mask_added = False

                if prompt_slice_idx in prompt_masks:
                    # Validate that we have objects with positive values in prompt masks
                    valid_objects = []
                    for ann_obj_id, mask in prompt_masks[prompt_slice_idx].items():
                        if torch.any(mask > 0):
                            valid_objects.append(ann_obj_id)

                    if not valid_objects:
                        pbar.update()
                        continue

                    # Process each object in the prompt masks
                    for ann_obj_id in valid_objects:
                        # Get the mask and ensure it's properly formatted
                        mask = prompt_masks[prompt_slice_idx][ann_obj_id].to(device=GPUdevice)

                        # Ensure the mask is 2D by flattening all batch and channel dimensions
                        if mask.dim() > 2:
                            # If mask has shape like [1, 1, H, W] or [B, C, H, W], convert to [H, W]
                            if mask.dim() == 4:
                                mask = mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
                            elif mask.dim() == 3:
                                mask = mask.squeeze(0)  # Remove just one dimension

                        # Ensure mask is 2D at this point
                        if mask.dim() != 2:
                            mask = mask.reshape(mask.shape[-2], mask.shape[-1])  # Force reshape to 2D if needed

                        # Ensure mask is binary (0 or 1)
                        mask = (mask > 0).float()

                        try:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=prompt_slice_idx,
                                obj_id=ann_obj_id,
                                mask=mask,
                            )
                            mask_added = True
                        except Exception as e:
                            continue

                if not mask_added:
                    pbar.update()
                    continue

                # Prepare for propagation
                net.train_propagate_in_video_preflight(train_state)

                # Store all segment results
                video_segments = {}

                # Forward propagation from prompt slice to end
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(
                        train_state,
                        start_frame_idx=prompt_slice_idx,
                        max_frame_num_to_track=video_length - prompt_slice_idx,
                        reverse=False
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Reset state and re-add masks for backward propagation
                if prompt_slice_idx > 0:
                    # Reset the inference state
                    net.reset_state(train_state)
                    train_state = net.train_init_state(imgs_tensor=imgs_tensor)

                    # Re-add mask prompts for backward propagation
                    mask_added_backward = False
                    if prompt_slice_idx in prompt_masks:
                        for ann_obj_id in valid_objects:
                            mask = prompt_masks[prompt_slice_idx][ann_obj_id].to(device=GPUdevice)

                            # Ensure the mask is 2D
                            if mask.dim() > 2:
                                if mask.dim() == 4:
                                    mask = mask.squeeze(0).squeeze(0)
                                elif mask.dim() == 3:
                                    mask = mask.squeeze(0)

                            if mask.dim() != 2:
                                mask = mask.reshape(mask.shape[-2], mask.shape[-1])

                            mask = (mask > 0).float()

                            try:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=prompt_slice_idx,
                                    obj_id=ann_obj_id,
                                    mask=mask,
                                )
                                mask_added_backward = True
                            except Exception as e:
                                continue

                    if mask_added_backward:
                        # Prepare for backward propagation
                        net.train_propagate_in_video_preflight(train_state)

                        # Backward propagation from prompt slice to beginning
                        for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(
                                train_state,
                                start_frame_idx=prompt_slice_idx,
                                max_frame_num_to_track=prompt_slice_idx,
                                reverse=True
                        ):
                            video_segments[out_frame_idx] = {
                                out_obj_id: out_mask_logits[i]
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }

                # Calculate loss
                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0

                # Frames to evaluate
                frames_to_eval = list(range(min(video_length, len(imgs_tensor))))
                valid_frames = 0

                for frame_idx in frames_to_eval:
                    if frame_idx not in video_segments:
                        continue

                    for ann_obj_id in obj_list:
                        if ann_obj_id not in video_segments[frame_idx]:
                            continue

                        pred = video_segments[frame_idx][ann_obj_id]
                        pred = pred.unsqueeze(0)

                        try:
                            mask = mask_dict[frame_idx][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                            # Ensure mask has the same dimensionality as pred
                            if mask.dim() != pred.dim():
                                # If mask is [1, 1024, 1024] and pred is [1, 1, 1024, 1024]
                                if mask.dim() == 3 and pred.dim() == 4:
                                    mask = mask.unsqueeze(1)  # Add channel dimension
                                elif mask.dim() == 2 and pred.dim() == 4:
                                    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                        except KeyError:
                            # Create empty mask if this object doesn't exist in this frame
                            mask = torch.zeros_like(pred).to(device=GPUdevice)

                        # Get bounding box for current frame if available
                        current_bbox = gland_bboxes[frame_idx] if gland_bboxes is not None and frame_idx < len(
                                gland_bboxes) else None
                        # Visualization during training if enabled
                        if args.train_vis:
                            filename = pack['image_meta_dict']['filename_or_obj']
                            if isinstance(filename, list):
                                filename = filename[0]
                            os.makedirs(f'./temp/train/{filename}/{frame_idx}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[frame_idx, :3, :, :].detach().cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{filename}/{frame_idx}/{obj_list.index(ann_obj_id)}.png',
                                        bbox_inches='tight', pad_inches=0)
                            plt.close()

                        # Calculate loss
                        obj_loss = region_wise_bce_loss(pred, mask, loss_func=lossfunc, bbox=current_bbox, pos_weight=pos_weight) + paper_loss(pred, mask)
                        # yy = lossfunc(pred, mask) #lossfunc(pred, mask)
                        # np.save(f'pred_{epoch}_{batch_idx}_{frame_idx}.npy', pred.cpu().detach())
                        # np.save(f'mask_{epoch}_{batch_idx}_{frame_idx}.npy', mask.cpu().detach())

                        # import matplotlib.pyplot as plt
                        # fig, (ax1, ax2) = plt.subplots(1, 2)
                        # ax1.imshow(pred.cpu().detach()[0, 0])
                        # ax1.scatter(current_bbox[1], current_bbox[0])
                        # ax1.scatter(current_bbox[3], current_bbox[2])
                        # ax2.imshow(mask.cpu().detach()[0, 0])

                        loss += obj_loss.item()
                        valid_frames += 1

                        # Separate prompt slice loss and non-prompt slice loss
                        if frame_idx == prompt_slice_idx:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                # Normalize losses
                if valid_frames > 0:
                    loss = loss / valid_frames
                else:
                    loss = 0

                prompt_loss = prompt_loss / max(1, len(obj_list))  # Only the prompt frame

                non_prompt_frames = valid_frames - (1 if prompt_slice_idx in video_segments else 0)
                if non_prompt_frames > 0:
                    non_prompt_loss = non_prompt_loss / non_prompt_frames / len(obj_list)
                else:
                    non_prompt_loss = 0

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item() if isinstance(prompt_loss, torch.Tensor) else prompt_loss
                epoch_non_prompt_loss += non_prompt_loss.item() if isinstance(non_prompt_loss,
                                                                              torch.Tensor) else non_prompt_loss

                prompt_loss_tensor = prompt_loss if isinstance(prompt_loss, torch.Tensor) else torch.tensor(0.0, device=GPUdevice)
                non_prompt_loss_tensor = non_prompt_loss if isinstance(non_prompt_loss, torch.Tensor) else torch.tensor(0.0, device=GPUdevice)

                total_loss = (0.5 * prompt_loss_tensor) + (1.0 * non_prompt_loss_tensor)
                if total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                if total_loss <= 0:
                    print('?????hello????')
                # # Update memory layers if non-prompt loss exists
                # if isinstance(non_prompt_loss,
                #               torch.Tensor) and non_prompt_loss.requires_grad and optimizer2 is not None:
                #     non_prompt_loss.backward(retain_graph=True)
                #     optimizer2.step()

                # # Update SAM layers with prompt loss
                # if optimizer1 is not None and isinstance(prompt_loss, torch.Tensor) and prompt_loss.requires_grad:
                #     prompt_loss.backward()
                #     optimizer1.step()
                #     optimizer1.zero_grad()

                # if optimizer2 is not None:
                #     optimizer2.zero_grad()

                # Reset the state for next batch
                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(
        train_loader)


def validation_sam_with_mask_prompt(args, val_loader, epoch, net: nn.Module, writer=None):
    """
    Validation function for SAM model using mask prompts
    """
    # Evaluation mode
    net.eval()

    n_val = len(val_loader)
    mix_res = (0,) * 2  # (IOU, DICE)
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    lossfunc = criterion_G
    # lossfunc = paper_loss

    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label'][0]
            prompt_slice_idx = pack['prompt_slice_idx'][0].item()
            # Get gland bounding boxes if available
            gland_bboxes = pack['gland_bboxes'] if 'gland_bboxes' in pack else None

            # Get prompt masks
            prompt_masks = {}
            if 'prompt_masks' in pack:
                prompt_masks = pack['prompt_masks']
            else:
                if prompt_slice_idx in mask_dict:
                    prompt_masks[prompt_slice_idx] = mask_dict[prompt_slice_idx]

            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)

            # Initialize inference state
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)

            # Get list of all objects
            obj_list = []
            for id in mask_dict.keys():
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))

            if len(obj_list) == 0:
                continue

            with torch.no_grad():
                # Add mask prompt for each object at the prompt slice
                mask_added = False
                if prompt_slice_idx in prompt_masks:
                    # Validate that we have objects with positive values
                    valid_objects = []
                    for ann_obj_id in obj_list:
                        if ann_obj_id in prompt_masks[prompt_slice_idx]:
                            mask = prompt_masks[prompt_slice_idx][ann_obj_id].to(device=GPUdevice)
                            if torch.any(mask > 0):
                                valid_objects.append(ann_obj_id)

                    if not valid_objects:
                        pbar.update()
                        continue

                    # Process each valid object in the prompt masks
                    for ann_obj_id in valid_objects:
                        mask = prompt_masks[prompt_slice_idx][ann_obj_id].to(device=GPUdevice)

                        # Ensure the mask is 2D
                        if mask.dim() > 2:
                            if mask.dim() == 4:
                                mask = mask.squeeze(0).squeeze(0)
                            elif mask.dim() == 3:
                                mask = mask.squeeze(0)

                        if mask.dim() != 2:
                            mask = mask.reshape(mask.shape[-2], mask.shape[-1])

                        # Ensure mask is binary (0 or 1)
                        mask = (mask > 0).float()

                        try:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=prompt_slice_idx,
                                obj_id=ann_obj_id,
                                mask=mask,
                            )
                            mask_added = True
                        except Exception as e:
                            continue

                if not mask_added:
                    pbar.update()
                    continue

                # Prepare for bidirectional propagation
                net.propagate_in_video_preflight(train_state)

                # Store all segment results
                video_segments = {}

                # Forward propagation from prompt slice to end
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(
                        train_state,
                        start_frame_idx=prompt_slice_idx,
                        max_frame_num_to_track=len(range(imgs_tensor.size(0))) - prompt_slice_idx,
                        reverse=False
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Reset state and re-add masks for backward propagation
                if prompt_slice_idx > 0:
                    # Reset the inference state
                    net.reset_state(train_state)
                    train_state = net.val_init_state(imgs_tensor=imgs_tensor)

                    # Re-add mask prompts for backward propagation
                    mask_added_backward = False
                    if prompt_slice_idx in prompt_masks:
                        for ann_obj_id in valid_objects:
                            mask = prompt_masks[prompt_slice_idx][ann_obj_id].to(device=GPUdevice)

                            # Ensure the mask is 2D
                            if mask.dim() > 2:
                                if mask.dim() == 4:
                                    mask = mask.squeeze(0).squeeze(0)
                                elif mask.dim() == 3:
                                    mask = mask.squeeze(0)

                            if mask.dim() != 2:
                                mask = mask.reshape(mask.shape[-2], mask.shape[-1])

                            mask = (mask > 0).float()

                            try:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=prompt_slice_idx,
                                    obj_id=ann_obj_id,
                                    mask=mask,
                                )
                                mask_added_backward = True
                            except Exception as e:
                                continue

                    if mask_added_backward:
                        # Prepare for backward propagation
                        net.propagate_in_video_preflight(train_state)

                        # Backward propagation from prompt slice to beginning
                        for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(
                                train_state,
                                start_frame_idx=prompt_slice_idx,
                                max_frame_num_to_track=prompt_slice_idx,
                                reverse=True
                        ):
                            video_segments[out_frame_idx] = {
                                out_obj_id: out_mask_logits[i]
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }

                # Calculate metrics
                loss = 0
                pred_iou = 0
                pred_dice = 0
                total_frames = 0

                frame_ids = list(range(imgs_tensor.size(0)))
                for id in frame_ids:
                    if id not in video_segments:
                        continue

                    for ann_obj_id in obj_list:
                        if ann_obj_id not in video_segments[id]:
                            continue

                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)

                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                            if mask.dim() != pred.dim():
                               # Add missing dimensions to match pred
                                if mask.dim() == 3 and pred.dim() == 4:
                                    mask = mask.unsqueeze(1)
                                elif mask.dim() == 2 and pred.dim() == 4:
                                    mask = mask.unsqueeze(0).unsqueeze(0)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)

                        current_bbox = gland_bboxes[id] if gland_bboxes is not None and id < len(gland_bboxes) else None

                        # Visualization during validation if enabled
                        if args.vis:
                            filename = pack['image_meta_dict']['filename_or_obj'][0]
                            if isinstance(filename, list):
                                filename = filename[0]
                            os.makedirs(f'./temp/val/{filename}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :3, :, :].cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/val/{filename}/{id}/{ann_obj_id}.png', bbox_inches='tight',
                                        pad_inches=0)
                            plt.close()

                        # Calculate loss and metrics
                        # loss += lossfunc(pred, mask)
                        loss += region_wise_bce_loss(pred, mask, loss_func=lossfunc, bbox=current_bbox, pos_weight=pos_weight) + paper_loss(pred, mask)

                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]
                        total_frames += 1

                # Normalize metrics
                if total_frames > 0:
                    loss = loss / total_frames
                    temp = (pred_iou / total_frames, pred_dice / total_frames)
                    tot += loss
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            # Reset state for next batch
            net.reset_state(train_state)
            pbar.update()

    return tot / n_val, tuple([a / n_val for a in mix_res])

def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5

            with torch.cuda.amp.autocast():
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
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

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
