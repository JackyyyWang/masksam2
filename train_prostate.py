# train_prostate.py
#!/usr/bin/env python3

""" Train SAM2 for 3D medical image segmentation with mask prompting
    Modified for 3-channel data with bidirectional propagation
    -net sam2 -exp_name prostate -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -dataset prostate -data_path /home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres

    -net sam2 -exp_name prostate -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -dataset prostate_box -data_path /home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres
    -net sam2 -exp_name prostate -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -dataset prostate_box_isolated -data_path /home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres
"""

import os
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function_gland_boxes as function
# from func_3d import function as function

from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from multimodal_utils import modify_model_for_multimodal
from multi_channel_block import MultiChannelDiffusionBlock
from types import SimpleNamespace
import torch.nn.functional as F

from diffusion_fusion import LightweightDiffusionFusion


def train_with_diffusion(model, diffusion, optimizer, data_loader):
    for batch in data_loader:
        x_multimodal = batch['image'].to(dtype=torch.float32, device=next(model.parameters()).device)
        B = x_multimodal.shape[0]
        t = torch.randint(0, diffusion.timesteps, (B,), device=x_multimodal.device)
        x_noisy = diffusion.add_noise(x_multimodal, t)
        x_pred = diffusion.denoise(x_noisy, t)
        L_recon = F.mse_loss(x_pred, x_multimodal)
        struct_losses = diffusion.structure_preservation_loss(x_multimodal, x_pred)
        L_struct = struct_losses['total']
        fused = diffusion.fast_sample(x_multimodal)
        pred_masks = model(fused)['masks'] if isinstance(model(fused), dict) else model(fused)
        L_seg = pred_masks.mean()  # placeholder since segmentation_loss is external
        loss = L_seg + 0.1 * L_recon + 0.1 * L_struct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    args = cfg.parse_args()

    # Setup device and model
    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    # Modify model for 3-channel input (using your existing utility)
    # net = modify_model_for_multimodal(net, in_channels=3)

    # Attach diffusion-based fusion block for multimodal input
    diff_cfg = SimpleNamespace(num_modalities=3)
    fusion_block = MultiChannelDiffusionBlock(diff_cfg).to(device=GPUdevice)
    if hasattr(net, 'image_encoder'):
        net.image_encoder.fusion_block = fusion_block

    net.to(dtype=torch.bfloat16)

    # Load pretrained weights if specified
    if args.pretrain:
        print(f"Loading pretrained weights from {args.pretrain}")
        weights = torch.load(args.pretrain)

        # Handle potential key mismatches due to model modification
        if isinstance(weights, dict) and 'model' in weights:
            model_weights = weights['model']
            # Filter out keys that don't match
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in model_weights.items() if
                               k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
        else:
            # Try to load directly with strict=False to skip mismatched layers
            net.load_state_dict(weights, strict=False)

    # Setup optimizers for different layers
    sam_layers = (
            []
            + list(net.sam_mask_decoder.parameters())
    )
    mem_layers = (
            []
            + list(net.obj_ptr_proj.parameters())
            + list(net.memory_encoder.parameters())
            + list(net.memory_attention.parameters())
            + list(net.mask_downsample.parameters())
    )

    if len(sam_layers) == 0:
        optimizer1 = None
    else:
        optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    if len(mem_layers) == 0:
        optimizer2 = None
    else:
        optimizer2 = optim.Adam(mem_layers, lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Enable mixed precision training
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Enable TF32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup logging
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # Get data loaders - ensure the dataloader returns the mask prompt
    train_loader, test_loader = get_dataloader(args)

    # Optimizer for diffusion fusion
    diffusion = fusion_block.diffusion
    optimizer_diff = optim.Adam(diffusion.parameters(), lr=1e-4)

    # Setup checkpoint path and tensorboard
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))

    # Create checkpoint folder
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # Start training
    best_score = 0.0
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):
        # Train for one epoch
        net.train()
        time_start = time.time()

        # Stage 1: train diffusion fusion module
        train_with_diffusion(net, diffusion, optimizer_diff, train_loader)

        # Using mask prompting
        loss, prompt_loss, non_prompt_loss = function.train_sam_with_mask_prompt(
            args, net, optimizer1, optimizer2, train_loader, epoch
        )

        logger.info(
            f'Train loss: {loss}, Prompt loss: {prompt_loss}, Non-prompt loss: {non_prompt_loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('Time for training:', time_end - time_start)

        # Validate model
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            time_start = time.time()

            tol, (eiou, edice) = function.validation_sam_with_mask_prompt(args, test_loader, epoch, net, writer)

            logger.info(f'Validation - Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            time_end = time.time()
            print('Time for validation:', time_end - time_start)

            # Save latest model
            torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
            if epoch % 5 == 0:
                torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], f'epoch_{epoch}.pth'))

            # Save best model
            if edice > best_dice:
                best_dice = edice
                torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'best_dice.pth'))
                logger.info(f'New best DICE score: {best_dice} @ epoch {epoch}')

    writer.close()
    logger.info(f'Training completed. Best DICE score: {best_dice}')


if __name__ == '__main__':
    main()

