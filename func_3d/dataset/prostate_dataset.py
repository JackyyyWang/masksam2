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

        # Return appropriate dictionary based on prompt type
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx
            }
        elif self.prompt == 'mask':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'prompt_masks': prompt_masks,
                'image_meta_dict': image_meta_dict,
                'prompt_slice_idx': prompt_frame_idx
            }

if __name__ == '__main__':
    class Args:
        image_size = 128  # or 256, etc.
        video_length = 16


    from torch.utils.data import DataLoader
    import argparse

    args = Args()

    # Path to a test dataset directory with .npz files
    data_path = '/home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres'  # 🔁 update this path!

    # Try all three prompts one by one
    for prompt_type in ['mask', 'bbox', 'click']:
        print(f"\n--- Testing prompt type: {prompt_type} ---")

        dataset = Prostate3DDataset(
            args=args,
            data_path=data_path,
            mode='Training',
            prompt=prompt_type,
            seed=42,
            variation=0  # any small integer for bbox jittering
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Get a single batch
        for batch in dataloader:
            key_ = list(batch['prompt_masks'].keys())
            print(key_)
            print(batch['prompt_masks'][key_[0]][1].shape)
            print("Image shape:", batch['image'].shape)  # [1, video_length, 5, H, W]

            # Check keys depending on prompt type
            if prompt_type == 'mask':
                print("Prompt mask keys:", list(batch['prompt_masks'].keys()))
            elif prompt_type == 'bbox':
                print("Bounding box dict keys:", list(batch['bbox'].keys()))
            elif prompt_type == 'click':
                print("Point labels keys:", list(batch['p_label'].keys()))
                print("Points dict keys:", list(batch['pt'].keys()))

            print("Label dict keys:", list(batch['label'].keys()))
            print("Prompt slice index:", batch['prompt_slice_idx'])
            print("Image meta:", batch['image_meta_dict'])
            break  # Only one sample for test