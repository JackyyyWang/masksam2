""" Dataloader for 3D medical image segmentation with SAM2
    Modified for 5-channel input data (5 x Slice x H x W)
"""
""" Dataloader for 3D medical image segmentation with SAM2
    Modified for 5-channel input data (5 x Slice x H x W)
    With built-in train/val split functionality
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from PIL import Image

from func_3d.utils import random_click, generate_bbox
from tqdm import tqdm


def normalize_channels(volume_data):
    """
    Normalize each channel of the volume data independently to range 0-255

    Args:
        volume_data: Input data with shape [channels, slices, height, width]
                    (typically 3 or 5 channels)

    Returns:
        Normalized data with the same shape
    """
    # Create a copy to avoid modifying the original data
    normalized_data = np.zeros_like(volume_data, dtype=np.float32)

    # Normalize each channel independently
    for c in range(volume_data.shape[0]):
        channel_data = volume_data[c]

        # Get min and max values for this channel
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        # Handle the case where min equals max (constant values)
        if min_val == max_val:
            # Set to mid-range (127.5) to avoid division by zero
            normalized_data[c] = np.ones_like(channel_data) * 127.5
        else:
            # Normalize to 0-255 range
            normalized_data[c] = 255.0 * (channel_data - min_val) / (max_val - min_val)

    return normalized_data


def process_gland_masks_and_generate_boxes(volume_data, dilation_pixels=5):
    """
    Process gland masks from the volume data and generate dilated bounding boxes.

    Args:
        volume_data: Input data with shape [channels, slices, height, width]
        dilation_pixels: Number of pixels to dilate the bounding box

    Returns:
        combined_mask: Binary mask combining both gland channels
        bbox_mask: 3D binary mask with dilated bounding boxes
        bounding_boxes: List of bounding box coordinates for each slice
    """
    # Extract gland mask channels (last 2 channels)
    gland_mask_data = volume_data[3:]  # Now 2 x Slice x H x W

    # Get dimensions
    _, z_dim, height, width = gland_mask_data.shape

    # Initialize outputs
    combined_mask = np.zeros((z_dim, height, width), dtype=np.uint8)
    bbox_mask = np.zeros_like(combined_mask)
    bounding_boxes = []

    # Process each slice
    for z in range(z_dim):
        # Combine both gland mask channels with threshold = 1
        mask1 = gland_mask_data[0, z] >= 1
        mask2 = gland_mask_data[1, z] >= 1
        slice_combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)
        combined_mask[z] = slice_combined_mask

        # Skip empty slices
        if not np.any(slice_combined_mask):
            bounding_boxes.append(None)
            continue

        # Find all non-zero pixel coordinates for the combined mask
        y_indices, x_indices = np.where(slice_combined_mask > 0)

        if len(y_indices) > 0:
            # Get the min and max coordinates to form a bounding box
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)

            # Apply dilation to bounding box
            y_min = max(0, y_min - dilation_pixels)
            x_min = max(0, x_min - dilation_pixels)
            y_max = min(height - 1, y_max + dilation_pixels)
            x_max = min(width - 1, x_max + dilation_pixels)

            # Save bounding box coordinates
            bounding_boxes.append((x_min, y_min, x_max, y_max))

            # Create bounding box mask
            bbox_mask[z, y_min:y_max + 1, x_min:x_max + 1] = 1
        else:
            bounding_boxes.append(None)

    filled_bounding_boxes = fill_missing_bounding_boxes(bounding_boxes)
    # Update bbox_mask with filled bounding boxes
    for z in range(z_dim):
        if bounding_boxes[z] is None and filled_bounding_boxes[z] is not None:
            x_min, y_min, x_max, y_max = filled_bounding_boxes[z]
            bbox_mask[z, y_min:y_max + 1, x_min:x_max + 1] = 1

    return combined_mask, bbox_mask, bounding_boxes


# def fill_missing_bounding_boxes(bounding_boxes):
#     """
#     Fill None bounding boxes with data from adjacent slices.
#
#     Args:
#         bounding_boxes: List of bounding box coordinates or None
#
#     Returns:
#         filled_bounding_boxes: List of bounding box coordinates with None values filled
#     """
#     filled_bounding_boxes = bounding_boxes.copy()
#     z_dim = len(bounding_boxes)
#
#     # If all bounding boxes are None, return the original list
#     if all(box is None for box in bounding_boxes):
#         return filled_bounding_boxes
#
#     # First forward pass: fill None with the previous valid bounding box
#     last_valid_box = None
#     for z in range(z_dim):
#         if filled_bounding_boxes[z] is not None:
#             last_valid_box = filled_bounding_boxes[z]
#         elif last_valid_box is not None:
#             filled_bounding_boxes[z] = last_valid_box
#
#     # Backward pass: fill remaining None values with the next valid bounding box
#     last_valid_box = None
#     for z in range(z_dim - 1, -1, -1):
#         if filled_bounding_boxes[z] is not None:
#             last_valid_box = filled_bounding_boxes[z]
#         elif last_valid_box is not None:
#             filled_bounding_boxes[z] = last_valid_box
#
#     return filled_bounding_boxes

def fill_missing_bounding_boxes(bounding_boxes):
    """
    Fill None bounding boxes with data from adjacent slices or a default value.

    Args:
        bounding_boxes: List of bounding box coordinates or None

    Returns:
        filled_bounding_boxes: List of bounding box coordinates with no None values
    """
    filled_bounding_boxes = bounding_boxes.copy()
    z_dim = len(bounding_boxes)

    # If all bounding boxes are None, use a default value for all
    if all(box is None for box in bounding_boxes):
        # Default to a small box in the center
        default_box = (64, 64, 192, 192)  # Adjust based on your image size
        return [default_box] * z_dim

    # Forward pass: fill None with the previous valid bounding box
    last_valid_box = None
    for z in range(z_dim):
        if filled_bounding_boxes[z] is not None:
            last_valid_box = filled_bounding_boxes[z]
        elif last_valid_box is not None:
            filled_bounding_boxes[z] = last_valid_box

    # Backward pass: fill remaining None values with the next valid bounding box
    last_valid_box = None
    for z in range(z_dim - 1, -1, -1):
        if filled_bounding_boxes[z] is not None:
            last_valid_box = filled_bounding_boxes[z]
        elif last_valid_box is not None:
            filled_bounding_boxes[z] = last_valid_box

    return filled_bounding_boxes


def scale_bbox(bbox, orig_size, target_size):
    """
    Scale a bounding box from original image dimensions to target dimensions.

    Args:
        bbox: Tuple of (x_min, y_min, x_max, y_max)
        orig_size: Original image size (height, width)
        target_size: Target image size (height, width)

    Returns:
        Scaled bounding box coordinates
    """
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox

    # Calculate scaling factors
    scale_x = target_size[1] / orig_size[1]
    scale_y = target_size[0] / orig_size[0]

    # Scale the coordinates
    x_min_scaled = int(x_min * scale_x)
    y_min_scaled = int(y_min * scale_y)
    x_max_scaled = int(x_max * scale_x)
    y_max_scaled = int(y_max * scale_y)

    return (x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled)

class Prostate3DDataset(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training',
                 prompt='mask', seed=None, variation=0, fold=0, num_folds=5, split_ratio=0.8,
                 filter_empty=True):
        """
        Dataset for 3D medical image segmentation using SAM2

        Args:
            args: Arguments from argparse
            data_path: Path to the data directory
            transform: Transform for images
            transform_msk: Transform for masks
            mode: 'Training' or 'Validation'
            prompt: Type of prompt ('mask', 'click', or 'bbox')
            seed: Random seed
            variation: Variation for bbox generation
            fold: Current fold for cross-validation (default: 0)
            num_folds: Number of folds for cross-validation (default: 5)
            split_ratio: Train/val split ratio if not using cross-validation (default: 0.8)
            filter_empty: Whether to filter out cases with no segmentation (default: True)
        """
        # Set the data list
        all_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

        # Sort files for reproducibility
        all_files.sort()

        # Set random seed for reproducible splits
        random_state = np.random.RandomState(seed if seed is not None else 42)

        # If filter_empty is True, preprocess to filter out empty cases
        if filter_empty:
            print("Filtering out cases with empty segmentations...")
            valid_files = []
            for file in tqdm(all_files, desc="Checking segmentations"):
                npz_path = os.path.join(data_path, file)
                try:
                    npz_data = np.load(npz_path)
                    seg = npz_data['seg']
                    # Check if segmentation has any positive values
                    if np.any(seg > 0):
                        valid_files.append(file)
                except Exception as e:
                    continue

            print(f"Filtered {len(all_files) - len(valid_files)} empty cases out of {len(all_files)} total")
            all_files = valid_files

        # If fold > 0, use k-fold cross-validation
        if num_folds > 1:
            # Shuffle files with a fixed seed for reproducibility
            indices = np.arange(len(all_files))
            random_state.shuffle(indices)

            # Compute fold size
            fold_size = len(all_files) // num_folds

            # Determine validation indices for the current fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(all_files)
            val_indices = indices[val_start:val_end]

            # Training indices are all indices not in validation
            train_indices = np.array([i for i in indices if i not in val_indices])

            # Select files for current mode
            if mode == 'Training':
                self.data_files = [all_files[i] for i in train_indices]
            else:  # Validation
                self.data_files = [all_files[i] for i in val_indices]
        else:
            # Use simple train/val split based on split_ratio
            num_files = len(all_files)
            indices = np.arange(num_files)
            random_state.shuffle(indices)

            split_idx = int(num_files * split_ratio)

            if mode == 'Training':
                self.data_files = [all_files[i] for i in indices[:split_idx]]
            else:  # Validation
                self.data_files = [all_files[i] for i in indices[split_idx:]]

        print(f"Created {mode} dataset with {len(self.data_files)} files")

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
            self.video_length = args.video_length if hasattr(args, 'video_length') else 16

    def sample_prompt_slice(self, seg_volume):
        """Sample a slice with segmentation as prompt"""
        # Handle segmentation shape - adjust for potential 1 x Slice x H x W format
        if seg_volume.ndim == 4 and seg_volume.shape[0] == 1:
            seg_volume = seg_volume[0]  # Convert to Slice x H x W

        # Make sure we're working with a 3D volume
        if seg_volume.ndim == 2:  # If we somehow got a single slice
            seg_volume = seg_volume[np.newaxis, :, :]  # Add slice dimension

        # Find slices that have segmentation and calculate segmentation areas
        valid_slices = []
        slice_areas = []

        # Scan through all slices
        for z in range(seg_volume.shape[0]):
            seg_area = np.sum(seg_volume[z] > 0)
            if seg_area > 0:
                valid_slices.append(z)
                slice_areas.append(seg_area)

        # If after all this we still don't have valid slices (should never happen with filtering)
        if not valid_slices:
            # Return the middle slice as a fallback
            return seg_volume.shape[0] // 2

        # Multiple sampling strategies
        if self.mode == 'Training':
            # For training - use weighted random sampling based on segmentation area
            # This gives preference to slices with larger lesions
            weights = np.array(slice_areas) / sum(slice_areas)
            chosen_idx = np.random.choice(len(valid_slices), p=weights)
            return valid_slices[chosen_idx]
        else:
            # For validation - use the slice with the largest segmentation area
            max_area_idx = np.argmax(slice_areas)
            return valid_slices[max_area_idx]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        """Get a 3D volume and process it for training"""
        # Load npz file
        npz_path = os.path.join(self.data_path, self.data_files[index])
        npz_data = np.load(npz_path)

        # Extract data and segmentation
        volume_data = npz_data['data']  # 5-channel data
        volume_seg = npz_data['seg']  # Segmentation masks

        combined_gland_mask, gland_bbox_mask, gland_bboxes = process_gland_masks_and_generate_boxes(
            volume_data, dilation_pixels=15
        )

        # Use only first 3 channels (as RGB)
        volume_data = volume_data[:3]  # Now 3 x Slice x H x W
        volume_data = normalize_channels(volume_data)
        # Ensure correct dimensions for segmentation
        if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
            volume_seg = volume_seg[0]  # Convert to Slice x H x W

        # Sample a slice with segmentation to use as prompt
        prompt_slice_idx = self.sample_prompt_slice(volume_seg)

        # Get dimensions
        n_channels = volume_data.shape[0]
        z_dim = volume_data.shape[1]

        # Determine actual video length based on volume size
        if self.video_length is None or self.video_length > z_dim:
            video_length = z_dim
        else:
            video_length = self.video_length

        # Find start frame ensuring we have complete sequence
        # And ensuring prompt_slice_idx is included in the range
        if z_dim > video_length and self.mode == 'Training':
            # Make sure prompt_slice_idx will be included in our range
            max_start = min(z_dim - video_length, prompt_slice_idx)
            min_start = max(0, prompt_slice_idx - video_length + 1)

            if max_start >= min_start:
                starting_frame = np.random.randint(min_start, max_start + 1)
            else:
                starting_frame = min_start
        else:
            starting_frame = 0

        # Adjust prompt_slice_idx to be relative to starting_frame
        prompt_frame_idx = prompt_slice_idx - starting_frame

        # Ensure prompt_frame_idx is within range
        if prompt_frame_idx < 0 or prompt_frame_idx >= video_length:
            # Adjust starting_frame to ensure prompt_slice_idx is included
            if prompt_slice_idx < video_length:
                starting_frame = 0
            else:
                starting_frame = prompt_slice_idx - video_length + 1
            prompt_frame_idx = prompt_slice_idx - starting_frame

        # Prepare tensors for images and masks
        img_tensor = torch.zeros(video_length, n_channels, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # Process each slice in our video
        for z_idx in range(starting_frame, starting_frame + video_length):
            if z_idx >= z_dim:
                break

            # Get slice data and segmentation
            slice_data = np.transpose(volume_data[:, z_idx], (1, 2, 0))  # [H, W, 3]
            slice_seg = volume_seg[z_idx]

            # Get unique object IDs in segmentation (non-zero values)
            obj_list = np.unique(slice_seg[slice_seg > 0])
            diff_obj_mask_dict = {}

            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}

            # Process each object in the segmentation
            for obj_id in obj_list:
                obj_mask = slice_seg == obj_id

                # Skip if mask is empty
                if not np.any(obj_mask):
                    continue

                # Resize mask using PIL for proper resizing
                from PIL import Image
                obj_mask_pil = Image.fromarray((obj_mask * 255).astype(np.uint8))
                obj_mask_resized = obj_mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)

                # Convert back to binary mask tensor
                obj_mask_np = np.array(obj_mask_resized) > 0
                obj_mask_tensor = torch.tensor(obj_mask_np).unsqueeze(0).int()

                # Store mask
                diff_obj_mask_dict[int(obj_id)] = obj_mask_tensor

                # Only create prompts for the selected prompt slice
                if z_idx == prompt_slice_idx:
                    if self.prompt == 'click':
                        diff_obj_point_label_dict[int(obj_id)], diff_obj_pt_dict[int(obj_id)] = random_click(
                            np.array(obj_mask_tensor.squeeze(0)), 1, seed=self.seed
                        )
                    elif self.prompt == 'bbox':
                        diff_obj_bbox_dict[int(obj_id)] = generate_bbox(
                            np.array(obj_mask_tensor.squeeze(0)), variation=self.variation, seed=self.seed
                        )

            # Resize and prepare image data
            slice_tensor = torch.from_numpy(slice_data).permute(2, 0, 1).float()  # [3, H, W]

            # Resize image if needed
            if slice_tensor.shape[1] != self.img_size or slice_tensor.shape[2] != self.img_size:
                resized_tensor = F.interpolate(
                    slice_tensor.unsqueeze(0),  # Add batch dimension [1, 3, H, W]
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension [3, img_size, img_size]
            else:
                resized_tensor = slice_tensor

            # Store in the output tensor
            frame_idx = z_idx - starting_frame
            img_tensor[frame_idx] = resized_tensor

            # Store masks if they exist
            if diff_obj_mask_dict:
                mask_dict[frame_idx] = diff_obj_mask_dict

            # Store prompt-related data if on prompt slice
            if z_idx == prompt_slice_idx:
                if self.prompt == 'bbox' and diff_obj_bbox_dict:
                    bbox_dict[frame_idx] = diff_obj_bbox_dict
                elif self.prompt == 'click' and diff_obj_pt_dict:
                    pt_dict[frame_idx] = diff_obj_pt_dict
                    point_label_dict[frame_idx] = diff_obj_point_label_dict

        # Prepare metadata
        image_meta_dict = {'filename_or_obj': self.data_files[index]}

        # Create prompt masks dictionary for mask-based prompting
        prompt_masks = {}

        if self.prompt == 'mask':
            # Check if prompt frame has masks
            if prompt_frame_idx in mask_dict and mask_dict[prompt_frame_idx]:
                prompt_masks[prompt_frame_idx] = mask_dict[prompt_frame_idx]
            else:
                # Look for any frame with masks as fallback
                for f_idx, masks in mask_dict.items():
                    if masks:  # If there are any objects in this frame
                        prompt_masks[f_idx] = masks
                        # Update prompt_frame_idx to match our fallback
                        prompt_frame_idx = f_idx
                        break

        # If we still have no valid prompt masks after all this filtering and fallbacks
        if self.prompt == 'mask' and not prompt_masks:
            # Choose a different sample as fallback
            fallback_index = (index + 1) % len(self.data_files)
            return self.__getitem__(fallback_index)

        gland_bboxes_aligned = gland_bboxes[starting_frame:starting_frame + video_length]

        # Add scaling for bounding boxes
        orig_height, orig_width = volume_data.shape[2], volume_data.shape[3]  # Original dimensions
        scaled_bboxes = []

        for bbox in gland_bboxes_aligned:
            scaled_bbox = scale_bbox(bbox, (orig_height, orig_width), (self.img_size, self.img_size))
            scaled_bboxes.append(scaled_bbox)

        # Replace the unscaled bounding boxes with scaled ones
        gland_bboxes_aligned = scaled_bboxes

        # Return appropriate dictionary based on prompt type
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx,
                'gland_bboxes': gland_bboxes_aligned
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx,
                'gland_bboxes': gland_bboxes_aligned
            }
        elif self.prompt == 'mask':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'prompt_masks': prompt_masks,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx,
                'gland_bboxes': gland_bboxes_aligned
            }


def check_segmentation_within_bbox(dataset, num_samples=None):
    """
    Check whether all segmentation masks are properly contained within their
    corresponding gland bounding boxes.

    Args:
        dataset: The Prostate3DDataset instance
        num_samples: Number of samples to check (None for all samples)

    Returns:
        results_summary: Dictionary with statistics and problem cases
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    # Sample indices to check
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    results = {
        'total_samples': num_samples,
        'total_slices_checked': 0,
        'slices_with_segmentation': 0,
        'slices_with_bbox': 0,
        'segmentation_outside_bbox': 0,
        'problem_cases': []
    }

    print(f"Checking {num_samples} samples for segmentation-bbox alignment...")

    for sample_idx in tqdm(indices):
        sample = dataset[sample_idx]

        # Extract data
        label_dict = sample['label']  # Dictionary of masks
        gland_bboxes = sample['gland_bboxes']  # Bounding boxes for glands
        filename = sample['image_meta_dict']['filename_or_obj']

        # Check each slice in the sample
        for slice_idx in range(len(gland_bboxes)):
            results['total_slices_checked'] += 1

            # Check if this slice has segmentation
            has_segmentation = slice_idx in label_dict and len(label_dict[slice_idx]) > 0

            if has_segmentation:
                results['slices_with_segmentation'] += 1

            # Check if this slice has a bounding box
            has_bbox = gland_bboxes[slice_idx] is not None

            if has_bbox:
                results['slices_with_bbox'] += 1

            # If both segmentation and bbox exist, check if segmentation is within bbox
            if has_segmentation and has_bbox:
                bbox = gland_bboxes[slice_idx]
                x_min, y_min, x_max, y_max = bbox

                # Check each object in the segmentation
                for obj_id, mask_tensor in label_dict[slice_idx].items():
                    mask = mask_tensor.squeeze().numpy()

                    # Find pixels where mask is non-zero
                    mask_y, mask_x = np.where(mask > 0)

                    # Check if any mask pixels are outside the bounding box
                    outside_bbox = False

                    for i in range(len(mask_y)):
                        if (mask_x[i] < x_min or mask_x[i] > x_max or
                                mask_y[i] < y_min or mask_y[i] > y_max):
                            outside_bbox = True
                            break

                    if outside_bbox:
                        results['segmentation_outside_bbox'] += 1
                        results['problem_cases'].append({
                            'filename': filename,
                            'sample_idx': sample_idx,
                            'slice_idx': slice_idx,
                            'obj_id': obj_id,
                            'bbox': bbox
                        })

    # Calculate percentages
    total_seg_and_bbox = results['slices_with_segmentation']
    if total_seg_and_bbox > 0:
        error_percentage = (results['segmentation_outside_bbox'] / total_seg_and_bbox) * 100
        results['error_percentage'] = error_percentage
    else:
        results['error_percentage'] = 0

    print("\n=== Segmentation-Bounding Box Alignment Check Results ===")
    print(f"Total samples checked: {results['total_samples']}")
    print(f"Total slices checked: {results['total_slices_checked']}")
    print(f"Slices with segmentation: {results['slices_with_segmentation']}")
    print(f"Slices with bounding boxes: {results['slices_with_bbox']}")
    print(f"Slices with segmentation outside bounding box: {results['segmentation_outside_bbox']}")
    print(f"Error percentage: {results['error_percentage']:.2f}%")

    if results['problem_cases']:
        print("\nProblem cases:")
        for i, case in enumerate(results['problem_cases'][:10]):  # Show first 10 problem cases
            print(f"  {i + 1}. File: {case['filename']}, Slice: {case['slice_idx']}, Object: {case['obj_id']}")

        if len(results['problem_cases']) > 10:
            print(f"  ... and {len(results['problem_cases']) - 10} more")

    return results

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader
    import argparse
    import matplotlib.patches as patches
    from tqdm import tqdm


    class Args:
        image_size = 256  # Higher resolution for better visualization
        video_length = 4


    args = Args()

    # Path to your dataset directory with .npz files
    data_path = '/home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres'  # Update this path!

    # Create output directory for visualizations
    vis_output_dir = 'visualization_results'
    os.makedirs(vis_output_dir, exist_ok=True)

    # Create dataset - only testing 'mask' prompt
    print("\n--- Testing with mask prompting ---")
    dataset = Prostate3DDataset(
        args=args,
        data_path=data_path,
        mode='Training',
        prompt='mask',
        seed=42
    )

    # Set number of samples to visualize
    num_samples = 3

    # Sample randomly from the dataset
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for sample_idx in sample_indices:
        print(f"\nProcessing sample {sample_idx}")
        sample = dataset[sample_idx]

        # Extract important data
        image_tensor = sample['image'][:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # [video_length, 3, H, W]
        label_dict = sample['label']  # Dictionary of masks
        prompt_masks = sample['prompt_masks']  # Prompt masks
        prompt_slice_idx = sample['prompt_slice_idx']  # Index of prompt slice
        gland_bboxes = sample['gland_bboxes']  # Bounding boxes for glands

        # Create a directory for this sample
        sample_dir = os.path.join(vis_output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)

        # Get filename for reference
        filename = sample['image_meta_dict']['filename_or_obj']

        # 1. Visualize the full video sequence with bounding boxes
        print("Visualizing full video sequence...")
        for frame_idx in range(len(image_tensor)):
            # Get current frame
            frame = image_tensor[frame_idx].permute(1, 2, 0).numpy() / 255.0

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(frame)

            # Add bounding box if available
            if gland_bboxes[frame_idx] is not None:
                x_min, y_min, x_max, y_max = gland_bboxes[frame_idx]
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)

            # Add segmentation mask overlay if available
            if frame_idx in label_dict:
                for obj_id, mask_tensor in label_dict[frame_idx].items():
                    mask = mask_tensor.squeeze().numpy()
                    # Create a semi-transparent overlay for the mask
                    mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                    mask_overlay[mask > 0] = [1, 0, 0, 0.3]  # Semi-transparent red
                    ax.imshow(mask_overlay)

            # Highlight if this is the prompt slice
            title = f"Frame {frame_idx}"
            if frame_idx == prompt_slice_idx:
                title += " (Prompt Slice)"
                plt.title(title, fontsize=16, color='red', fontweight='bold')
            else:
                plt.title(title, fontsize=14)

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'frame_{frame_idx:03d}.png'), dpi=150)
            plt.close()

        # 2. Create a special visualization for the prompt slice
        if prompt_slice_idx is not None:
            print("Creating detailed prompt slice visualization...")
            # Get prompt frame
            prompt_frame = image_tensor[prompt_slice_idx].permute(1, 2, 0).numpy() / 255.0

            # Create a 2x2 grid for visualization
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))

            # Original image
            axs[0, 0].imshow(prompt_frame)
            axs[0, 0].set_title('Original Image', fontsize=14)
            axs[0, 0].axis('off')

            # Prompt mask visualization
            if prompt_slice_idx in prompt_masks:
                combined_mask = np.zeros((args.image_size, args.image_size), dtype=np.float32)
                for obj_id, mask_tensor in prompt_masks[prompt_slice_idx].items():
                    mask = mask_tensor.squeeze().numpy()
                    combined_mask += mask * obj_id  # Use obj_id as intensity for different objects

                axs[0, 1].imshow(combined_mask, cmap='viridis')
                axs[0, 1].set_title('Prompt Masks', fontsize=14)
                axs[0, 1].axis('off')
            else:
                axs[0, 1].text(0.5, 0.5, 'No prompt masks available',
                               horizontalalignment='center', verticalalignment='center')
                axs[0, 1].axis('off')

            # Overlay of segmentation on image
            axs[1, 0].imshow(prompt_frame)

            if prompt_slice_idx in label_dict:
                for obj_id, mask_tensor in label_dict[prompt_slice_idx].items():
                    mask = mask_tensor.squeeze().numpy()
                    # Create a semi-transparent overlay
                    mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                    mask_overlay[mask > 0] = [1, 0, 0, 0.5]  # Semi-transparent red
                    axs[1, 0].imshow(mask_overlay)

            axs[1, 0].set_title('Segmentation Overlay', fontsize=14)
            axs[1, 0].axis('off')

            # Bounding box visualization
            axs[1, 1].imshow(prompt_frame)

            if gland_bboxes[prompt_slice_idx] is not None:
                x_min, y_min, x_max, y_max = gland_bboxes[prompt_slice_idx]
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=3, edgecolor='g', facecolor='none'
                )
                axs[1, 1].add_patch(rect)
                axs[1, 1].set_title('Gland Bounding Box', fontsize=14)
            else:
                axs[1, 1].set_title('No Gland Bounding Box Available', fontsize=14)

            axs[1, 1].axis('off')

            plt.suptitle(f"Prompt Slice Details (Frame {prompt_slice_idx})\nFile: {filename}", fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'prompt_slice_details.png'), dpi=200)
            plt.close()

        # 3. Create a video sequence montage (a grid of frames)
        print("Creating video sequence montage...")
        rows = 4
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

        for i in range(rows):
            for j in range(cols):
                frame_idx = i * cols + j
                if frame_idx < len(image_tensor):
                    # Get current frame
                    frame = image_tensor[frame_idx].permute(1, 2, 0).numpy() / 255.0

                    axs[i, j].imshow(frame)

                    # Add bounding box if available
                    if gland_bboxes[frame_idx] is not None:
                        x_min, y_min, x_max, y_max = gland_bboxes[frame_idx]
                        rect = patches.Rectangle(
                            (x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        axs[i, j].add_patch(rect)

                    # Highlight if this is the prompt slice
                    if frame_idx == prompt_slice_idx:
                        axs[i, j].set_title(f"Frame {frame_idx} (Prompt)", fontsize=12, color='red')
                    else:
                        axs[i, j].set_title(f"Frame {frame_idx}", fontsize=10)

                    axs[i, j].axis('off')
                else:
                    axs[i, j].axis('off')

        plt.suptitle(f"Video Sequence Overview\nFile: {filename}", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'video_sequence_montage.png'), dpi=200)
        plt.close()

        # 4. Create GIF for the sequence (optional, requires PIL)
        try:
            from PIL import Image as PILImage

            print("Creating GIF animation...")
            frames = []

            for frame_idx in range(len(image_tensor)):
                # Create temporary figure
                fig, ax = plt.subplots(figsize=(10, 10))

                # Get current frame
                frame = image_tensor[frame_idx].permute(1, 2, 0).numpy() / 255.0
                ax.imshow(frame)

                # Add bounding box if available
                if gland_bboxes[frame_idx] is not None:
                    x_min, y_min, x_max, y_max = gland_bboxes[frame_idx]
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)

                # Add segmentation mask overlay if available
                if frame_idx in label_dict:
                    for obj_id, mask_tensor in label_dict[frame_idx].items():
                        mask = mask_tensor.squeeze().numpy()
                        # Create a semi-transparent overlay for the mask
                        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                        mask_overlay[mask > 0] = [1, 0, 0, 0.3]  # Semi-transparent red
                        ax.imshow(mask_overlay)

                # Highlight if this is the prompt slice
                title = f"Frame {frame_idx}"
                if frame_idx == prompt_slice_idx:
                    title += " (Prompt Slice)"
                    ax.set_title(title, fontsize=16, color='red', fontweight='bold')
                else:
                    ax.set_title(title, fontsize=14)

                ax.axis('off')
                plt.tight_layout()

                # Save to a temporary file
                temp_file = os.path.join(sample_dir, 'temp.png')
                plt.savefig(temp_file, dpi=100)
                plt.close()

                # Open and append to frames
                img = PILImage.open(temp_file)
                frames.append(img.copy())
                img.close()

            # Save as GIF
            if frames:
                frames[0].save(
                    os.path.join(sample_dir, 'animation.gif'),
                    save_all=True,
                    append_images=frames[1:],
                    duration=300,  # milliseconds per frame
                    loop=0
                )

            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except ImportError:
            print("PIL package not available. Skipping GIF creation.")

        print(f"Visualization for sample {sample_idx} complete!")

    print("\nVisualization complete! Results saved to:", vis_output_dir)

    # Add this after the visualization part:
    print("\n\n=== Running Segmentation-Bounding Box Alignment Check ===")
    alignment_results = check_segmentation_within_bbox(dataset, num_samples=1000)  # Check 50 random samples

    # Save alignment check results
    import json

    with open(os.path.join(vis_output_dir, 'alignment_check_results.json'), 'w') as f:
        # Convert numpy int64 to int for JSON serialization
        alignment_results_serializable = json.loads(
            json.dumps(alignment_results, default=lambda o: int(o) if isinstance(o, np.int64) else o))
        json.dump(alignment_results_serializable, f, indent=4)

    # If there are problem cases, create visualizations for the first few
    if alignment_results['problem_cases']:
        problem_dir = os.path.join(vis_output_dir, 'problem_cases')
        os.makedirs(problem_dir, exist_ok=True)

        print("\nCreating visualizations for problem cases...")
        num_to_visualize = min(5, len(alignment_results['problem_cases']))

        for i in range(num_to_visualize):
            case = alignment_results['problem_cases'][i]
            sample_idx = case['sample_idx']
            slice_idx = case['slice_idx']
            obj_id = case['obj_id']

            # Get the sample data
            sample = dataset[sample_idx]
            image_tensor = sample['image']
            label_dict = sample['label']
            bbox = case['bbox']

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 10))

            # Display the image
            frame = image_tensor[slice_idx].permute(1, 2, 0).numpy() / 255.0
            ax.imshow(frame)

            # Draw the bounding box
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Draw the mask
            mask = label_dict[slice_idx][obj_id].squeeze().numpy()

            # Find pixels where mask is non-zero
            mask_y, mask_x = np.where(mask > 0)

            # Plot mask pixels
            ax.plot(mask_x, mask_y, 'g.', markersize=1)

            # Highlight pixels outside the box in a different color
            outside_x = []
            outside_y = []
            for j in range(len(mask_x)):
                if (mask_x[j] < x_min or mask_x[j] > x_max or
                        mask_y[j] < y_min or mask_y[j] > y_max):
                    outside_x.append(mask_x[j])
                    outside_y.append(mask_y[j])

            if outside_x:
                ax.plot(outside_x, outside_y, 'b.', markersize=3)

            # Set title and save
            plt.title(f"Problem Case: File {case['filename']}\nSlice {slice_idx}, Object {obj_id}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(problem_dir, f'problem_case_{i + 1}.png'), dpi=200)
            plt.close()

        print(f"Visualizations for {num_to_visualize} problem cases saved to {problem_dir}")

    print("\nAll tests complete!")


