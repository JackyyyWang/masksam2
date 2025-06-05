import os
import csv
import blosc2
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import ndimage
from func_3d.utils import get_network
from multimodal_utils import modify_model_for_multimodal
import cv2

# --- Config ---
# Dataset911_Prostate3Test Dataset103_Basel2 Dataset104_HT Dataset105_UKSH Dataset106_OHSU Dataset107_Loyola Dataset108_UCLA Dataset109_Prostate158
# Dataset701_NewSitesPirads4 Dataset702_NewSitesPirads5 Dataset703_NewSitesPirads3

data_root = "/data/pct_ids_nvme/users/z005257c/nnUNet_preprocessed/Dataset703_NewSitesPirads3/nnUNetPlans_3d_fullres"
nifti_root = "/data/pct_ids_nvme/users/z005257c/nnUNet_raw/Dataset703_NewSitesPirads3/imagesTr"  # For affine information
case_mapping_path = "/data/pct_ids_nvme/users/z005257c/nnUNet_raw/Dataset703_NewSitesPirads3/case_mapping.csv"
output_root = "sam2_infer_703"


ckpt_path = "/data/pct_ids_nvme/users/z005257c/logs/prostate_2025_05_21_16_38_31/Model/best_dice.pth"
sam_ckpt = "/pct_ids/users/z005257c/GitHub/MaskSam2/checkpoints/sam2_hiera_small.pt"
img_size = 1024  # For SAM2 inference only
video_length = 16
use_gpu = True


def sample_prompt_slice(seg):
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim == 2:
        seg = seg[np.newaxis, :, :]
    valid_slices = []
    slice_areas = []
    for z in range(seg.shape[0]):
        area = np.sum(seg[z] > 0)
        if area > 0:
            valid_slices.append(z)
            slice_areas.append(area)
    return valid_slices[np.argmax(slice_areas)] if valid_slices else seg.shape[0] // 2


def find_3d_connected_components(volume_seg):
    """
    Find 3D connected components in a volume segmentation.
    """
    binary_mask = volume_seg > 0
    labeled_volume, num_components = ndimage.label(binary_mask, structure=np.ones((3, 3, 3)))
    return labeled_volume, num_components


def apply_adaptive_thresholding(pred):
    """Apply adaptive thresholding to prediction logits"""
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.array(pred)
    if pred_np.ndim == 4:
        logits = pred_np[0, 0]
    elif pred_np.ndim == 3:
        logits = pred_np[0]
    else:
        logits = pred_np

    if np.min(logits) == -1024.0 and np.max(logits) == -1024.0:
        return np.zeros_like(logits)

    pred_min = logits.min()
    pred_max = logits.max()
    pred_mean = logits.mean()
    pred_std = logits.std()

    if pred_max > 3.0:
        threshold = 0.0
    elif pred_max - pred_min > 5.0:
        threshold = pred_min + (pred_max - pred_min) * 0.75
    elif pred_std > 0.5:
        threshold = pred_mean + pred_std
    else:
        return np.zeros_like(logits)

    binary_mask = (logits > threshold)
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled_mask.flatten())
        if len(component_sizes) > 1:
            component_sizes = component_sizes[1:]
            largest_component = np.argmax(component_sizes) + 1
            min_size = max(50, int(component_sizes[largest_component - 1] * 0.1))
            large_enough = np.isin(labeled_mask, np.where(component_sizes >= min_size)[0] + 1)
            binary_mask = large_enough
    return binary_mask.astype(np.uint8)


def compute_dice(pred, target):
    """Compute dice coefficient"""
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def load_affine_from_nifti(case_id, nifti_root):
    """Load affine transformation from original NIFTI file"""
    nifti_path = os.path.join(nifti_root, f"prostate_{case_id}_0001.nii.gz")
    try:
        nifti_img = nib.load(nifti_path)
        return nifti_img.affine
    except FileNotFoundError:
        print(f"Warning: Could not find NIFTI file at {nifti_path}, using identity affine")
        return np.eye(4)


def save_nifti_results(imgs_data, ground_truth, predictions, prompt_slice_data, site, pid, output_dir, 
                      affine, component_id=None, start_idx=0, prompt_idx=0):
    """
    Save results as NIFTI files at original resolution with proper affine
    
    Args:
        imgs_data: Image data [channels, frames, height, width] at ORIGINAL resolution
        ground_truth: Ground truth masks [frames, height, width] at ORIGINAL resolution  
        predictions: Predicted masks [frames, height, width] at ORIGINAL resolution
        prompt_slice_data: Dict with prompt slice info
        site: Site identifier
        pid: Patient ID
        output_dir: Base output directory
        affine: 4x4 affine transformation matrix
        component_id: Component ID (if processing individual components)
        start_idx: Starting frame index in the full volume
        prompt_idx: Prompt frame index in the full volume
    """
    # Create folder structure
    if component_id is not None:
        case_dir = os.path.join(output_dir, f"site_{site}", pid, f"component_{component_id}")
    else:
        case_dir = os.path.join(output_dir, f"site_{site}", pid)
    os.makedirs(case_dir, exist_ok=True)

    # Rotate data 90 degrees to match NIFTI orientation
    def rotate_data(data):
        if data.ndim == 3:  # [frames, height, width]
            return np.stack([np.flip(np.rot90(slice), axis=1) for slice in data], axis=0)
        elif data.ndim == 4:  # [channels, frames, height, width]
            return np.stack([np.stack([np.flip(np.rot90(slice), axis=1) for slice in channel], axis=0)
                             for channel in data], axis=0)
        return data

    # Rotate the data
    imgs_data = rotate_data(imgs_data)
    ground_truth = rotate_data(ground_truth)
    predictions = rotate_data(predictions)

    imgs_data = np.transpose(imgs_data, (0, 2, 3, 1))

    ground_truth = np.transpose(ground_truth, (1, 2, 0))
    ground_truth = np.flip(np.rot90(np.flip(np.rot90(ground_truth), axis=1)), axis=0)

    predictions = np.transpose(predictions, (1, 2, 0))
    predictions = np.flip(np.rot90(np.flip(np.rot90(predictions), axis=1)), axis=0)
    
    # Save channel-specific NIFTI files
    for channel in range(imgs_data.shape[0]):
        channel_data = imgs_data[channel]
        channel_data = np.flip(np.rot90(np.flip(np.rot90(channel_data), axis=1)), axis=0)

        nifti_img = nib.Nifti1Image(channel_data, affine)
        nib.save(nifti_img, os.path.join(case_dir, f'channel_{channel}.nii.gz'))

    # Save ground truth segmentation
    gt_nifti = nib.Nifti1Image(ground_truth.astype(np.uint8), affine)
    nib.save(gt_nifti, os.path.join(case_dir, 'ground_truth.nii.gz'))

    # Save prediction
    pred_nifti = nib.Nifti1Image(predictions.astype(np.uint8), affine)
    nib.save(pred_nifti, os.path.join(case_dir, 'prediction.nii.gz'))

    # Save prompt slice information
    if prompt_slice_data:        
        # Save prompt mask
        prompt_mask = np.zeros((ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2]))
       
        prompt_mask[:, :, prompt_idx] = ground_truth[:, :, prompt_idx]
            
        prompt_mask_nifti = nib.Nifti1Image(prompt_mask.astype(np.uint8), affine)
        nib.save(prompt_mask_nifti, os.path.join(case_dir, 'prompt_slice_mask.nii.gz'))
        
        # Save prompt info as text
        with open(os.path.join(case_dir, 'prompt_info.txt'), 'w') as f:
            f.write(f"Prompt slice index (global): {prompt_idx}\n")
            f.write(f"Video start index: {start_idx}\n")
            f.write(f"Video end index: {start_idx + imgs_data.shape[1] - 1}\n")

    # Create and save FN, FP, TP map (1=FN, 2=FP, 3=TP)
    evaluation_map = np.zeros_like(ground_truth, dtype=np.uint8)

    # True Positives: Both ground truth and prediction are positive
    tp_mask = (ground_truth > 0) & (predictions > 0)
    evaluation_map[tp_mask] = 3

    # False Positives: Prediction is positive but ground truth is negative
    fp_mask = (ground_truth == 0) & (predictions > 0)
    evaluation_map[fp_mask] = 2

    # False Negatives: Ground truth is positive but prediction is negative
    fn_mask = (ground_truth > 0) & (predictions == 0)
    evaluation_map[fn_mask] = 1

    # Save evaluation map
    eval_nifti = nib.Nifti1Image(evaluation_map, affine)
    nib.save(eval_nifti, os.path.join(case_dir, 'evaluation_map.nii.gz'))

    # Calculate metrics at ORIGINAL resolution
    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Save metrics as text file
    with open(os.path.join(case_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Dice Score: {dice:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"Total Pixels in Ground Truth: {np.sum(ground_truth > 0)}\n")
        f.write(f"Total Pixels in Prediction: {np.sum(predictions > 0)}\n")
        f.write(f"Image Resolution: {ground_truth.shape[1]} x {ground_truth.shape[2]}\n")
        f.write(f"Number of Frames: {ground_truth.shape[0]}\n")

    return dice


def load_model():
    class Args:
        net = 'sam2'
        sam_config = 'sam2_hiera_s'
        image_size = img_size
        gpu = use_gpu
        gpu_device = 0

    args = Args()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    args.sam_ckpt = sam_ckpt
    net = get_network(args, args.net, use_gpu=use_gpu, gpu_device=device)
    net = modify_model_for_multimodal(net, in_channels=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    net.to(device)
    net.eval()
    return net, device


def run():
    os.makedirs(output_root, exist_ok=True)
    model, device = load_model()
    all_dice_scores = []
    component_dice_scores = []

    with open(case_mapping_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            site = row['site']
            case = row['case']
            case_id = row['case_id']
            pid = row.get('pid', f"PID_{case}")
            case_name = f"prostate{site}_{case}"
            print(f"Processing {case_name}...")

            # Load affine information from original NIFTI
            affine = load_affine_from_nifti(case_id, nifti_root)

            # Load data and segmentation
            # data = blosc2.open(urlpath=os.path.join(data_root, f"prostate_{case_id}.b2nd"), mode='r', mmap_mode='r')
            # seg = blosc2.open(urlpath=os.path.join(data_root, f"prostate_{case_id}_seg.b2nd"), mode='r', mmap_mode='r')
            # data = np.asarray(data[:3])  # Take first 3 channels
            # seg = np.asarray(seg)
            data = np.load(os.path.join(data_root, f"prostate_{case_id}.npz"))['data'][:3]
            seg = np.load(os.path.join(data_root, f"prostate_{case_id}.npz"))['seg']
            if seg.ndim == 4 and seg.shape[0] == 1:
                seg = seg[0]

            # Get original image dimensions
            original_shape = (seg.shape[1], seg.shape[2])  # H, W
            print(f"Original image shape: {original_shape}")

            # Find connected components in the original segmentation
            labeled_components, num_components = find_3d_connected_components(seg)
            print(f"Found {num_components} connected components for {case_name}")

            # If there are multiple components, process each separately
            components_to_process = range(1, num_components + 1) if num_components > 0 else [0]

            # Keep track of overall predictions for the case (at original resolution)
            overall_predictions = np.zeros_like(seg)

            for comp_id in components_to_process:
                print(f"Processing component {comp_id} of {num_components} for {case_name}")

                # Create mask for just this component
                if num_components > 0:
                    component_mask = labeled_components == comp_id
                    component_seg = np.zeros_like(seg)
                    component_seg[component_mask] = 1
                else:
                    component_seg = seg
                    component_mask = seg > 0

                # Sample a prompt slice from this component
                prompt_idx = sample_prompt_slice(component_seg)

                # Determine video range
                video_length_actual = data.shape[1]
                start_idx = max(0, min(prompt_idx - video_length_actual // 2, data.shape[1] - video_length_actual))
                
                # Extract video segment
                imgs = data[:, start_idx:start_idx + video_length_actual]
                masks = component_seg[start_idx:start_idx + video_length_actual]

                # Convert images to tensor and resize to 1024x1024 for SAM2 inference ONLY
                imgs_tensor = torch.from_numpy(imgs).float().to(device)
                imgs_tensor = F.interpolate(
                    imgs_tensor.unsqueeze(0),
                    size=(video_length_actual, img_size, img_size),
                    mode='trilinear',
                    align_corners=False
                )[0]
                imgs_tensor = imgs_tensor.permute(1, 0, 2, 3)

                # Initialize model state
                train_state = model.val_init_state(imgs_tensor=imgs_tensor)

                # Skip if no objects in this component
                if not np.any(component_mask):
                    print(f"No segmentation found for component {comp_id}, skipping")
                    continue

                # Create prompt mask for this component at the prompt slice (resize to 1024 for SAM2)
                mask = torch.tensor((component_seg[prompt_idx] > 0).astype(np.float32))
                mask = F.interpolate(mask[None, None], size=(img_size, img_size), mode='nearest')[0, 0].to(device)

                # Add mask to model state
                model.train_add_new_mask(train_state, frame_idx=prompt_idx - start_idx, obj_id=1, mask=mask)
                model.propagate_in_video_preflight(train_state)

                # Propagate mask through video
                pred_masks = {t: {} for t in range(video_length_actual)}
                for reverse in [False, True]:
                    for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                            train_state, start_frame_idx=prompt_idx - start_idx,
                            max_frame_num_to_track=video_length_actual, reverse=reverse):
                        pred_masks[out_frame_idx] = {obj_id: out_mask_logits[i] for i, obj_id in enumerate(out_obj_ids)}

                # Prepare arrays for NIFTI output at ORIGINAL resolution
                predictions_orig = np.zeros((video_length_actual, original_shape[0], original_shape[1]), dtype=np.uint8)
                ground_truth_orig = np.zeros((video_length_actual, original_shape[0], original_shape[1]), dtype=np.uint8)
                imgs_data_orig = np.zeros((3, video_length_actual, original_shape[0], original_shape[1]), dtype=np.float32)

                dice_scores = []

                for t in range(video_length_actual):
                    # Process image data for all 3 channels at original resolution
                    for c in range(3):
                        img_c_orig = imgs[c, t]  # Original resolution image
                        # Normalize
                        p1, p99 = np.percentile(img_c_orig, (1, 99))
                        img_c_orig = np.clip((img_c_orig - p1) / (p99 - p1 + 1e-5), 0, 1)
                        imgs_data_orig[c, t] = img_c_orig

                    # Ground truth mask at original resolution
                    gt_mask_orig = (masks[t] > 0).astype(np.uint8)
                    ground_truth_orig[t] = gt_mask_orig

                    # Process prediction mask - resize from 1024 back to original resolution
                    pred = pred_masks[t].get(1)
                    if pred is not None:
                        # Apply thresholding at 1024 resolution
                        pred_mask_1024 = apply_adaptive_thresholding(pred)
                        # Resize back to original resolution
                        pred_mask_orig = cv2.resize(
                            pred_mask_1024.astype(np.uint8),
                            (original_shape[1], original_shape[0]),  # Width, Height
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        pred_mask_orig = np.zeros(original_shape, dtype=np.uint8)

                    predictions_orig[t] = pred_mask_orig

                    # Update overall predictions for the case
                    frame_idx = start_idx + t
                    if frame_idx < overall_predictions.shape[0]:
                        overall_predictions[frame_idx] = np.logical_or(
                            overall_predictions[frame_idx],
                            pred_mask_orig > 0
                        ).astype(np.uint8)

                    # Compute Dice score at original resolution
                    dice = compute_dice(pred_mask_orig, gt_mask_orig)
                    dice_scores.append(dice)

                # Prepare prompt slice data
                prompt_slice_data = {
                    'prompt_idx': prompt_idx,
                    'start_idx': start_idx,
                    'component_id': comp_id
                }

                # Save NIFTI files for this component at original resolution
                if num_components > 1:
                    component_dice = save_nifti_results(
                        imgs_data=imgs_data_orig,
                        ground_truth=ground_truth_orig,
                        predictions=predictions_orig,
                        prompt_slice_data=prompt_slice_data,
                        site=site,
                        pid=pid,
                        output_dir=output_root,
                        affine=affine,
                        component_id=comp_id,
                        start_idx=start_idx,
                        prompt_idx=prompt_idx
                    )

                    mean_dice = np.mean(dice_scores)
                    print(f"{case_name} Component {comp_id} Dice: {mean_dice:.4f}")
                    component_dice_scores.append({
                        "case": case_name,
                        "site": site,
                        "pid": pid,
                        "component_id": comp_id,
                        "dice": mean_dice
                    })
                else:
                    nifti_dice = save_nifti_results(
                        imgs_data=imgs_data_orig,
                        ground_truth=ground_truth_orig,
                        predictions=predictions_orig,
                        prompt_slice_data=prompt_slice_data,
                        site=site,
                        pid=pid,
                        output_dir=output_root,
                        affine=affine,
                        start_idx=start_idx,
                        prompt_idx=prompt_idx
                    )

                    mean_dice = np.mean(dice_scores)
                    print(f"{case_name} Dice: {mean_dice:.4f}")
                    all_dice_scores.append({"case": case_name, "site": site, "pid": pid, "dice": mean_dice})

            # For cases with multiple components, also save the combined result
            if num_components > 1:
                # Prepare combined data at original resolution
                video_length_actual = data.shape[1]
                
                # Process full volume at original resolution
                imgs_data_combined = np.zeros((3, video_length_actual, original_shape[0], original_shape[1]), dtype=np.float32)
                ground_truth_combined = np.zeros((video_length_actual, original_shape[0], original_shape[1]), dtype=np.uint8)

                for t in range(video_length_actual):
                    # Process image data at original resolution
                    for c in range(3):
                        img_c_orig = data[c, t]
                        p1, p99 = np.percentile(img_c_orig, (1, 99))
                        img_c_orig = np.clip((img_c_orig - p1) / (p99 - p1 + 1e-5), 0, 1)
                        imgs_data_combined[c, t] = img_c_orig

                    # Ground truth at original resolution
                    ground_truth_combined[t] = (seg[t] > 0).astype(np.uint8)

                # Use the accumulated overall_predictions
                predictions_combined = overall_predictions.astype(np.uint8)

                # Save the combined NIFTI
                prompt_slice_data_combined = {
                    'prompt_idx': 'multiple',
                    'start_idx': 0,
                    'component_id': 'combined'
                }
                
                combined_dice = save_nifti_results(
                    imgs_data=imgs_data_combined,
                    ground_truth=ground_truth_combined,
                    predictions=predictions_combined,
                    prompt_slice_data=prompt_slice_data_combined,
                    site=site,
                    pid=pid,
                    output_dir=output_root,
                    affine=affine,
                    component_id="combined"
                )

                # Calculate overall Dice score at original resolution
                overall_dice = compute_dice(predictions_combined, ground_truth_combined)

                print(f"{case_name} Combined Dice: {overall_dice:.4f}")
                all_dice_scores.append({
                    "case": case_name,
                    "site": site,
                    "pid": pid,
                    "dice": overall_dice
                })

    # Save all dice scores
    with open(os.path.join(output_root, "dice_scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "site", "pid", "dice"])
        writer.writeheader()
        writer.writerows(all_dice_scores)

    # Save component-specific dice scores
    if component_dice_scores:
        with open(os.path.join(output_root, "component_dice_scores.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "site", "pid", "component_id", "dice"])
            writer.writeheader()
            writer.writerows(component_dice_scores)

    # Create site-specific summary
    site_metrics = {}
    for metric in all_dice_scores:
        site = metric['site']
        if site not in site_metrics:
            site_metrics[site] = {
                'dice_scores': [],
                'cases': 0
            }
        site_metrics[site]['dice_scores'].append(metric['dice'])
        site_metrics[site]['cases'] += 1

    with open(os.path.join(output_root, "site_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["site", "avg_dice", "num_cases"])
        writer.writeheader()
        for site, metrics in site_metrics.items():
            writer.writerow({
                'site': site,
                'avg_dice': np.mean(metrics['dice_scores']),
                'num_cases': metrics['cases']
            })

    print("\nSummary by site:")
    for site, metrics in site_metrics.items():
        print(f"Site {site}: Dice = {np.mean(metrics['dice_scores']):.4f}, Cases = {metrics['cases']}")

    print(f"\nOverall Average Dice: {np.mean([score['dice'] for score in all_dice_scores]):.4f}")
    print(f"Total cases processed: {len(all_dice_scores)}")


if __name__ == '__main__':
    run()