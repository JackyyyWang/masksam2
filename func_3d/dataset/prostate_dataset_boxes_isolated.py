""" Dataloader for 3D medical image segmentation with SAM2
    Modified for 5-channel input data (5 x Slice x H x W)
    With built-in train/val split functionality
    Enhanced to treat connected components as separate samples
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from PIL import Image
from scipy import ndimage
from copy import deepcopy

from func_3d.utils import random_click, generate_bbox
from tqdm import tqdm
import time

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

    return combined_mask, bbox_mask, filled_bounding_boxes


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


def find_3d_connected_components(volume_seg):
    """
    Find 3D connected components in a volume segmentation.

    Args:
        volume_seg: 3D segmentation mask (Slice x H x W)

    Returns:
        components: Volume with labeled connected components
        num_components: Number of connected components found
    """
    # Ensure binary mask
    binary_mask = volume_seg > 0

    # Find 3D connected components (26-connectivity)
    labeled_volume, num_components = ndimage.label(binary_mask, structure=np.ones((3, 3, 3)))

    return labeled_volume, num_components


class Prostate3DDatasetWithConnectedComponents(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training',
                 prompt='mask', seed=None, variation=0, fold=0, num_folds=5, split_ratio=0.8,
                 filter_empty=True, min_component_slices=1):
        """
        Enhanced dataset for 3D medical image segmentation using SAM2 that treats
        connected components as separate samples.

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
            min_component_slices: Minimum number of slices a component must span to be considered valid
        """
        # Set random seed for reproducibility
        self.random_state = np.random.RandomState(seed if seed is not None else 42)
        self.min_component_slices = min_component_slices

        # Store the basic information of the dataset
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
        # Set the data list
        all_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

        # Sort files for reproducibility
        all_files.sort()

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
            self.random_state.shuffle(indices)

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
                self.data_files_original = [all_files[i] for i in train_indices]
            else:  # Validation
                self.data_files_original = [all_files[i] for i in val_indices]
        else:
            # Use simple train/val split based on split_ratio
            num_files = len(all_files)
            indices = np.arange(num_files)
            self.random_state.shuffle(indices)

            split_idx = int(num_files * split_ratio)

            if mode == 'Training':
                self.data_files_original = [all_files[i] for i in indices[:split_idx]]
            else:  # Validation
                self.data_files_original = [all_files[i] for i in indices[split_idx:]]

        print(f"Found {len(self.data_files_original)} base files for {mode} dataset")

        # Process each file to extract connected components and create enhanced data list
        self.enhanced_data_list = self._process_files_for_connected_components()

        print(
            f"Created enhanced {mode} dataset with {len(self.enhanced_data_list)} samples after connected component extraction")

    def _process_files_for_connected_components(self):
        """
        Process all files to extract connected components and create enhanced data list.

        Returns:
            enhanced_data_list: List of dictionaries containing file info and component info
        """
        enhanced_data_list = []

        print("Processing files to extract connected components...")
        for file_idx, filename in enumerate(tqdm(self.data_files_original, desc="Extracting components")):
            npz_path = os.path.join(self.data_path, filename)
            try:
                # Load data
                npz_data = np.load(npz_path)
                volume_seg = npz_data['seg']

                # Ensure correct dimensions for segmentation
                if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
                    volume_seg = volume_seg[0]  # Convert to Slice x H x W

                # Find 3D connected components
                labeled_components, num_components = find_3d_connected_components(volume_seg)

                if num_components == 0:
                    # No components found, skip this file
                    continue

                # If only one component, add the file as is
                if num_components == 1:
                    enhanced_data_list.append({
                        'filename': filename,
                        'component_id': 1,
                        'total_components': 1
                    })
                else:
                    # Multiple components found
                    # Check each component's slice coverage
                    for comp_id in range(1, num_components + 1):
                        # Create a mask for this component
                        component_mask = labeled_components == comp_id

                        # Count slices that contain this component
                        slices_with_component = 0
                        for z in range(component_mask.shape[0]):
                            if np.any(component_mask[z]):
                                slices_with_component += 1

                        # Only include components that span at least min_component_slices
                        if slices_with_component >= self.min_component_slices:
                            enhanced_data_list.append({
                                'filename': filename,
                                'component_id': comp_id,
                                'total_components': num_components
                            })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        return enhanced_data_list

    def sample_prompt_slice(self, component_mask, random_state):
        """
        Sample a slice with segmentation as prompt for a specific component.

        Args:
            component_mask: Binary mask for the specific component

        Returns:
            slice_idx: Index of the selected prompt slice
        """
        # Make sure we're working with a 3D volume
        if component_mask.ndim == 2:  # If we somehow got a single slice
            component_mask = component_mask[np.newaxis, :, :]  # Add slice dimension

        # Find slices that have segmentation and calculate segmentation areas
        valid_slices = []
        slice_areas = []

        # Scan through all slices
        for z in range(component_mask.shape[0]):
            seg_area = np.sum(component_mask[z] > 0)
            if seg_area > 0:
                valid_slices.append(z)
                slice_areas.append(seg_area)

        # If after all this we still don't have valid slices (should never happen with filtering)
        if not valid_slices:
            # Return the middle slice as a fallback
            return component_mask.shape[0] // 2

        # Multiple sampling strategies
        if self.mode == 'Training':
            # For training - use weighted random sampling based on segmentation area
            # This gives preference to slices with larger lesions
            weights = np.array(slice_areas) / sum(slice_areas)
            chosen_idx = random_state.choice(len(valid_slices), p=weights)
            return valid_slices[chosen_idx]
        else:
            # For validation - use the slice with the largest segmentation area
            max_area_idx = np.argmax(slice_areas)
            return valid_slices[max_area_idx]

    def __len__(self):
        return len(self.enhanced_data_list)

    def __getitem__(self, index):
        """Get a 3D volume with a specific component and process it for training"""
        # Get the enhanced data entry
        data_entry = self.enhanced_data_list[index]
        filename = data_entry['filename']
        component_id = data_entry['component_id']

        # Load npz file
        npz_path = os.path.join(self.data_path, filename)
        npz_data = np.load(npz_path)

        # Extract data and segmentation
        volume_data = npz_data['data']  # 5-channel data
        volume_seg = npz_data['seg']  # Segmentation masks

        # Process gland masks and bounding boxes (common for all components)
        combined_gland_mask, gland_bbox_mask, gland_bboxes = process_gland_masks_and_generate_boxes(
            volume_data, dilation_pixels=15
        )

        # Use only first 3 channels (as RGB)
        volume_data = volume_data[:3]  # Now 3 x Slice x H x W
        volume_data = normalize_channels(volume_data)

        # Ensure correct dimensions for segmentation
        if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
            volume_seg = volume_seg[0]  # Convert to Slice x H x W

        # Extract the specific component if needed
        if data_entry['total_components'] > 1:
            # Find 3D connected components
            labeled_components, _ = find_3d_connected_components(volume_seg)

            # Create a mask for just this component
            component_mask = labeled_components == component_id

            # Create a new segmentation volume with only this component
            component_volume_seg = np.zeros_like(volume_seg)
            component_volume_seg[component_mask] = 1
        else:
            # Only one component, use the original segmentation
            component_volume_seg = volume_seg.copy()
            component_mask = component_volume_seg > 0

        # Sample a slice with segmentation to use as prompt for this component
        call_random_state = np.random.RandomState(
            int(time.time() * 1000) % 10000 + index
        )
        prompt_slice_idx = self.sample_prompt_slice(component_mask, call_random_state)

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
            slice_seg = component_volume_seg[z_idx]

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
        image_meta_dict = {
            'filename_or_obj': filename,
            'component_id': component_id,
            'total_components': data_entry['total_components']
        }

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
            fallback_index = (index + 1) % len(self.enhanced_data_list)
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

    # Add to the dataset class
    def get_component_mask(self, volume_seg, component_id):
        """
        Extract the specific component mask from a segmentation volume.
        Intended for debugging/visualization purposes.

        Args:
            volume_seg: Original segmentation volume (Slice x H x W)
            component_id: ID of the component to extract

        Returns:
            component_mask: Binary mask for just this component
        """
        # Find 3D connected components
        labeled_components, _ = find_3d_connected_components(volume_seg)

        # Create a mask for just this component
        component_mask = labeled_components == component_id

        return component_mask


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
        component_id = sample['image_meta_dict'].get('component_id', 1)

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
                            'component_id': component_id,
                            'sample_idx': sample_idx,
                            'slice_idx': slice_idx,
                            'obj_id': obj_id,
                            'bbox': bbox
                        })


def validate_connected_components_dataset(data_path, output_dir='validation_results', num_samples=5):
    """
    Validate the Prostate3DDatasetWithConnectedComponents by comparing it with the original dataset
    and visualizing examples of isolated components.

    Args:
        data_path: Path to the npz files
        output_dir: Directory to save validation results
        num_samples: Number of samples to visualize
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import json
    import random
    from prostate_dataset_boxes import Prostate3DDataset

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    class Args:
        image_size = 256
        video_length = 8

    args = Args()

    print("Creating datasets...")
    # Create original dataset
    original_dataset = Prostate3DDataset(
        args=args,
        data_path=data_path,
        mode='Training',
        prompt='mask',
        seed=42
    )

    # Create enhanced dataset with connected components
    enhanced_dataset = Prostate3DDatasetWithConnectedComponents(
        args=args,
        data_path=data_path,
        mode='Training',
        prompt='mask',
        seed=42,
        min_component_slices=3  # Minimum number of slices a component should span
    )

    # Compare dataset sizes
    print(f"\nOriginal dataset size: {len(original_dataset)}")
    print(f"Enhanced dataset size: {len(enhanced_dataset)}")
    expansion_factor = len(enhanced_dataset) / len(original_dataset) if len(original_dataset) > 0 else 0
    print(f"Dataset expansion factor: {expansion_factor:.2f}x")

    # Create a mapping from original files to their components in the enhanced dataset
    file_to_components = {}
    for i, entry in enumerate(enhanced_dataset.enhanced_data_list):
        filename = entry['filename']
        comp_id = entry['component_id']
        if filename not in file_to_components:
            file_to_components[filename] = []
        file_to_components[filename].append((i, comp_id, entry['total_components']))

    # Find files with multiple components for visualization
    multi_component_files = {filename: comps for filename, comps in file_to_components.items()
                             if len(comps) > 1}

    print(f"Files with multiple components: {len(multi_component_files)}")

    # Save statistics to JSON
    statistics = {
        "original_dataset_size": len(original_dataset),
        "enhanced_dataset_size": len(enhanced_dataset),
        "expansion_factor": float(expansion_factor),
        "files_with_multiple_components": len(multi_component_files),
        "multi_component_file_examples": [
            {"filename": filename, "component_count": len(comps)}
            for filename, comps in list(multi_component_files.items())[:10]
        ]
    }

    with open(os.path.join(output_dir, "dataset_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)

    # Visualize samples from multi-component files
    if multi_component_files:
        print("\nVisualizing examples from files with multiple components...")

        # Select a random subset of multi-component files to visualize
        selected_files = random.sample(list(multi_component_files.keys()),
                                       min(num_samples, len(multi_component_files)))

        for file_idx, filename in enumerate(selected_files):
            components = multi_component_files[filename]
            print(f"  Visualizing {filename} with {len(components)} components")

            # Create a directory for this file
            file_dir = os.path.join(output_dir, f"file_{file_idx + 1}_{os.path.splitext(filename)[0]}")
            os.makedirs(file_dir, exist_ok=True)

            # Get the original sample first (for comparison)
            orig_idx = original_dataset.data_files.index(filename)
            original_sample = original_dataset[orig_idx]

            # Create a visualization for the original sample
            orig_image = original_sample['image']
            orig_label_dict = original_sample['label']
            orig_prompt_slice_idx = original_sample['prompt_slice_idx']

            # Find a good slice with segmentation for the original
            orig_slices_with_masks = sorted(list(orig_label_dict.keys()))
            if not orig_slices_with_masks:
                print(f"  No masks found in original sample for {filename}, skipping")
                continue

            # Create a figure with the original sample
            fig, ax = plt.subplots(figsize=(10, 10))

            # Choose a slice with good segmentation
            slice_idx = orig_prompt_slice_idx if orig_prompt_slice_idx in orig_label_dict else orig_slices_with_masks[0]

            # Display image
            frame = orig_image[slice_idx].permute(1, 2, 0).numpy() / 255.0
            ax.imshow(frame)

            # Add all segmentation masks with different colors
            cmap = plt.cm.get_cmap('tab10', len(orig_label_dict[slice_idx]))
            for i, (obj_id, mask_tensor) in enumerate(orig_label_dict[slice_idx].items()):
                mask = mask_tensor.squeeze().numpy()
                color = cmap(i)
                mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                mask_overlay[mask > 0] = [color[0], color[1], color[2], 0.5]
                ax.imshow(mask_overlay)

            ax.set_title(f"Original Sample - All Components\nSlice {slice_idx}", fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(file_dir, "original_sample.png"), dpi=200)
            plt.close()

            # Now visualize each component separately
            for comp_idx, (sample_idx, comp_id, total_comps) in enumerate(components):
                component_sample = enhanced_dataset[sample_idx]

                # Get data for this component
                comp_image = component_sample['image']
                comp_label_dict = component_sample['label']
                comp_prompt_slice_idx = component_sample['prompt_slice_idx']

                # Find slices with masks
                slices_with_masks = sorted(list(comp_label_dict.keys()))
                if not slices_with_masks:
                    print(f"  No masks found for component {comp_id}, skipping")
                    continue

                # Create a figure showing the component
                fig, axs = plt.subplots(1, 2, figsize=(20, 10))

                # Choose the same slice as the original if possible, otherwise use prompt slice
                if slice_idx in comp_label_dict:
                    comp_slice_idx = slice_idx
                elif comp_prompt_slice_idx in comp_label_dict:
                    comp_slice_idx = comp_prompt_slice_idx
                else:
                    comp_slice_idx = slices_with_masks[0]

                # Display image
                frame = comp_image[comp_slice_idx].permute(1, 2, 0).numpy() / 255.0
                axs[0].imshow(frame)

                # Add the component's segmentation mask
                for obj_id, mask_tensor in comp_label_dict[comp_slice_idx].items():
                    mask = mask_tensor.squeeze().numpy()
                    mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                    mask_overlay[mask > 0] = [0, 1, 0, 0.5]  # Green for this component
                    axs[0].imshow(mask_overlay)

                axs[0].set_title(f"Component {comp_id} of {total_comps}\nSlice {comp_slice_idx}", fontsize=14)
                axs[0].axis('off')

                # Show 3D visualization by displaying multiple slices
                # Find key slices for this component
                key_slices = []
                if comp_prompt_slice_idx in comp_label_dict:
                    key_slices.append(comp_prompt_slice_idx)

                # Add more slices from the component
                remaining_slices = [s for s in slices_with_masks if s != comp_prompt_slice_idx]
                num_extra_slices = min(5, len(remaining_slices))
                if num_extra_slices > 0:
                    step = max(1, len(remaining_slices) // num_extra_slices)
                    key_slices.extend(remaining_slices[::step][:num_extra_slices])

                # Sort slices
                key_slices.sort()

                # Create a grid for 3D visualization
                grid_size = int(np.ceil(np.sqrt(len(key_slices))))
                grid_axes = []

                for idx, s_idx in enumerate(key_slices):
                    if idx >= grid_size * grid_size:
                        break

                    row = idx // grid_size
                    col = idx % grid_size

                    # Calculate position in the grid
                    ax_pos = [
                        0.05 + (col / grid_size) * 0.9,
                        0.05 + ((grid_size - 1 - row) / grid_size) * 0.9,
                        0.9 / grid_size * 0.9,
                        0.9 / grid_size * 0.9
                    ]

                    # Create subplot
                    sub_ax = fig.add_axes(ax_pos)
                    grid_axes.append(sub_ax)

                    # Display slice
                    s_frame = comp_image[s_idx].permute(1, 2, 0).numpy() / 255.0
                    sub_ax.imshow(s_frame)

                    # Add mask if available
                    if s_idx in comp_label_dict:
                        for obj_id, mask_tensor in comp_label_dict[s_idx].items():
                            mask = mask_tensor.squeeze().numpy()
                            mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                            mask_overlay[mask > 0] = [0, 1, 0, 0.5]
                            sub_ax.imshow(mask_overlay)

                    sub_ax.set_title(f"Slice {s_idx}", fontsize=10)
                    sub_ax.axis('off')

                # Hide the second main axis since we're using our custom grid
                axs[1].axis('off')
                axs[1].set_title("3D View of Component Across Slices", fontsize=14)

                plt.suptitle(f"Component {comp_id} of {total_comps} from {filename}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(file_dir, f"component_{comp_id}.png"), dpi=200)
                plt.close()

            # Create a comparison visualization showing all components
            if len(components) > 1:
                # Determine how many slices to show
                key_slice = slice_idx  # Use the same slice we used for the original

                # Create a figure with the original and all components
                fig, axs = plt.subplots(1, len(components) + 1, figsize=(5 * (len(components) + 1), 5))

                # Show original first
                frame = orig_image[key_slice].permute(1, 2, 0).numpy() / 255.0
                axs[0].imshow(frame)

                # Add all segmentation masks with different colors
                if key_slice in orig_label_dict:
                    cmap = plt.cm.get_cmap('tab10', len(orig_label_dict[key_slice]))
                    for i, (obj_id, mask_tensor) in enumerate(orig_label_dict[key_slice].items()):
                        mask = mask_tensor.squeeze().numpy()
                        color = cmap(i)
                        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                        mask_overlay[mask > 0] = [color[0], color[1], color[2], 0.5]
                        axs[0].imshow(mask_overlay)

                axs[0].set_title(f"Original (All Components)\nSlice {key_slice}", fontsize=12)
                axs[0].axis('off')

                # Now show each component
                for i, (sample_idx, comp_id, _) in enumerate(components):
                    component_sample = enhanced_dataset[sample_idx]
                    comp_image = component_sample['image']
                    comp_label_dict = component_sample['label']

                    # Display the same slice
                    frame = comp_image[key_slice].permute(1, 2, 0).numpy() / 255.0
                    axs[i + 1].imshow(frame)

                    # Add segmentation if available
                    if key_slice in comp_label_dict:
                        for obj_id, mask_tensor in comp_label_dict[key_slice].items():
                            mask = mask_tensor.squeeze().numpy()
                            mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                            # Use a consistent color based on component ID
                            color = plt.cm.get_cmap('tab10')(comp_id % 10)
                            mask_overlay[mask > 0] = [color[0], color[1], color[2], 0.5]
                            axs[i + 1].imshow(mask_overlay)

                    axs[i + 1].set_title(f"Component {comp_id}\nSlice {key_slice}", fontsize=12)
                    axs[i + 1].axis('off')

                plt.suptitle(f"Component Comparison - {filename}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(file_dir, "component_comparison.png"), dpi=200)
                plt.close()

    print(f"\nValidation complete! Results saved to {output_dir}")
    return {
        "original_size": len(original_dataset),
        "enhanced_size": len(enhanced_dataset),
        "expansion_factor": expansion_factor,
        "multi_component_files": len(multi_component_files)
    }


def visualize_component_frame_selection(data_path, output_dir='frame_selection_results', num_samples=3):
    """
    Create simple visualizations showing how frame selection differs between
    components from the same subject. Shows channel[1] with bounding boxes.

    Args:
        data_path: Path to the npz files
        output_dir: Directory to save visualization results
        num_samples: Number of subjects to visualize
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    from collections import defaultdict

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    class Args:
        image_size = 256
        video_length = 8

    args = Args()

    print("Creating enhanced dataset with connected components...")
    # Create enhanced dataset with connected components
    enhanced_dataset = Prostate3DDatasetWithConnectedComponents(
        args=args,
        data_path=data_path,
        mode='Training',
        prompt='mask',
        seed=42,
        min_component_slices=3
    )

    # Group samples by original file
    file_to_samples = defaultdict(list)
    for i, entry in enumerate(enhanced_dataset.enhanced_data_list):
        filename = entry['filename']
        comp_id = entry['component_id']
        file_to_samples[filename].append((i, comp_id))

    # Find files with multiple components
    multi_component_files = {f: samples for f, samples in file_to_samples.items() if len(samples) > 1}

    if not multi_component_files:
        print("No files with multiple components found. Cannot demonstrate different frame selection.")
        return False

    # Select random subjects with multiple components
    selected_files = random.sample(list(multi_component_files.keys()),
                                   min(num_samples, len(multi_component_files)))

    for file_idx, filename in enumerate(selected_files):
        components = multi_component_files[filename]
        print(f"Visualizing frame selection for {filename} with {len(components)} components")

        # Create a directory for this file
        file_dir = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}")
        os.makedirs(file_dir, exist_ok=True)

        # Create figure for all components from this file
        num_components = len(components)

        # Get sample for each component
        component_samples = []
        for sample_idx, comp_id in components:
            sample = enhanced_dataset[sample_idx]
            component_samples.append((sample, comp_id))

        # Extract frame information for all components
        frame_info = []
        for sample, comp_id in component_samples:
            prompt_idx = sample['prompt_slice_idx']
            starting_frame = sample['image_meta_dict'].get('starting_frame', 0)  # May need to add this to dataset
            frame_info.append({
                'comp_id': comp_id,
                'prompt_idx': prompt_idx,
                'starting_frame': starting_frame,
                'has_masks': sorted(list(sample['label'].keys()))
            })

        # Save frame selection information to text file
        with open(os.path.join(file_dir, "frame_selection_info.txt"), "w") as f:
            f.write(f"Frame Selection Analysis for {filename}\n")
            f.write(f"Total components: {num_components}\n\n")
            for info in frame_info:
                f.write(f"Component {info['comp_id']}:\n")
                f.write(f"  Prompt frame index: {info['prompt_idx']}\n")
                f.write(f"  Starting frame: {info['starting_frame']}\n")
                f.write(f"  Frames with masks: {info['has_masks']}\n\n")

        # Create visualization for each component
        for idx, (sample, comp_id) in enumerate(component_samples):
            # Extract data
            image_tensor = sample['image']
            label_dict = sample['label']
            prompt_slice_idx = sample['prompt_slice_idx']
            gland_bboxes = sample['gland_bboxes']

            # Calculate how many slices to show (all of them if <= 8, otherwise select key frames)
            num_frames = len(image_tensor)
            if num_frames <= 8:
                frames_to_show = list(range(num_frames))
            else:
                # Show prompt slice and some distributed frames
                frames_to_show = [prompt_slice_idx]
                remaining_frames = [i for i in range(num_frames) if i != prompt_slice_idx]

                # Select ~7 more frames evenly distributed
                step = max(1, len(remaining_frames) // 7)
                selected_frames = remaining_frames[::step][:7]
                frames_to_show.extend(selected_frames)
                frames_to_show.sort()

            # Create a grid to show all selected frames
            rows = int(np.ceil(len(frames_to_show) / 4))
            cols = min(4, len(frames_to_show))

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if rows == 1 and cols == 1:
                axs = np.array([[axs]])
            elif rows == 1:
                axs = np.array([axs])
            elif cols == 1:
                axs = np.array([[ax] for ax in axs])

            # Plot each selected frame
            for i, frame_idx in enumerate(frames_to_show):
                row = i // cols
                col = i % cols

                # Get the specific channel (channel[1])
                frame = image_tensor[frame_idx, 1].numpy()

                # Normalize for display
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                if frame_max > frame_min:
                    frame_normalized = (frame - frame_min) / (frame_max - frame_min)
                else:
                    frame_normalized = np.zeros_like(frame)

                # Display the image
                axs[row, col].imshow(frame_normalized, cmap='gray')

                # Add bounding box if available
                if gland_bboxes[frame_idx] is not None:
                    x_min, y_min, x_max, y_max = gland_bboxes[frame_idx]
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axs[row, col].add_patch(rect)

                # Add segmentation mask overlay if available
                if frame_idx in label_dict:
                    for obj_id, mask_tensor in label_dict[frame_idx].items():
                        mask = mask_tensor.squeeze().numpy()
                        # Create a semi-transparent overlay for the mask
                        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                        mask_overlay[mask > 0] = [0, 1, 0, 0.5]  # Semi-transparent green
                        axs[row, col].imshow(mask_overlay)

                # Add title to indicate prompt slice
                if frame_idx == prompt_slice_idx:
                    axs[row, col].set_title(f"Frame {frame_idx} (Prompt)", color='red', fontsize=12)
                else:
                    axs[row, col].set_title(f"Frame {frame_idx}", fontsize=10)

                axs[row, col].axis('off')

            # Hide any unused subplots
            for i in range(len(frames_to_show), rows * cols):
                row = i // cols
                col = i % cols
                axs[row, col].axis('off')

            plt.suptitle(f"Component {comp_id} Frame Selection\nFile: {filename}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(file_dir, f"component_{comp_id}_frames.png"), dpi=150)
            plt.close()

        # Create a comparison visualization showing prompt frames for all components
        fig, axs = plt.subplots(1, num_components, figsize=(num_components * 5, 5))
        if num_components == 1:
            axs = [axs]

        # Show prompt frame for each component
        for i, (sample, comp_id) in enumerate(component_samples):
            prompt_idx = sample['prompt_slice_idx']

            # Get channel 1 for the prompt frame
            frame = sample['image'][prompt_idx, 1].numpy()

            # Normalize for display
            frame_min = np.min(frame)
            frame_max = np.max(frame)
            if frame_max > frame_min:
                frame_normalized = (frame - frame_min) / (frame_max - frame_min)
            else:
                frame_normalized = np.zeros_like(frame)

            # Display image
            axs[i].imshow(frame_normalized, cmap='gray')

            # Add segmentation mask overlay if available
            if prompt_idx in sample['label']:
                for obj_id, mask_tensor in sample['label'][prompt_idx].items():
                    mask = mask_tensor.squeeze().numpy()
                    mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                    mask_overlay[mask > 0] = [0, 1, 0, 0.5]  # Semi-transparent green
                    axs[i].imshow(mask_overlay)

            # Add bounding box if available
            if sample['gland_bboxes'][prompt_idx] is not None:
                x_min, y_min, x_max, y_max = sample['gland_bboxes'][prompt_idx]
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                axs[i].add_patch(rect)

            axs[i].set_title(f"Component {comp_id}\nPrompt Frame: {prompt_idx}", fontsize=12)
            axs[i].axis('off')

        plt.suptitle(f"Prompt Frame Comparison\nFile: {filename}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, "prompt_frame_comparison.png"), dpi=150)
        plt.close()

    print(f"\nVisualization complete! Results saved to {output_dir}")
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the Prostate3D Dataset with Connected Components")
    parser.add_argument("--data_path", type=str, help="Path to the npz files")
    parser.add_argument("--output_dir", type=str, default="validation_results",
                        help="Directory to save validation results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")



    args = parser.parse_args()
    args.data_path = "/home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres"
    args.output_dir = "./"

    visualize_component_frame_selection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

    # print("\n=== Dataset Validation Results ===")
    # print(f"Original dataset size: {results['original_size']} samples")
    # print(f"Enhanced dataset size: {results['enhanced_size']} samples")
    # print(f"Expansion factor: {results['expansion_factor']:.2f}x")
    # print(f"Files with multiple components: {results['multi_component_files']}")