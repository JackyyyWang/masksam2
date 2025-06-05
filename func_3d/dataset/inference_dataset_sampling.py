# inference_dataset.py (Modified)
import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import torch.nn.functional as F
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

def find_3d_connected_components_and_best_slices(volume_seg):
    """
    Find 3D connected components in the volume segmentation and determine 
    the best slice (largest area) for each component.

    Args:
        volume_seg: 3D segmentation mask (Slice x H x W)

    Returns:
        components_info: List of dictionaries with component info including best slice
        total_components: Total number of 3D components
    """
    components_info = []
    
    # Ensure binary mask
    binary_mask = volume_seg > 0
    
    if not np.any(binary_mask):
        return components_info, 0
    
    # Find 3D connected components (26-connectivity)
    labeled_volume, num_components = ndimage.label(binary_mask, structure=np.ones((3, 3, 3)))
    
    print(f"Found {num_components} 3D connected components")
    
    for comp_id in range(1, num_components + 1):
        # Get the 3D mask for this component
        component_3d_mask = labeled_volume == comp_id
        total_volume = np.sum(component_3d_mask)
        
        # Find which slices contain this component
        slices_with_component = []
        slice_areas = {}
        
        for z in range(volume_seg.shape[0]):
            slice_mask = component_3d_mask[z]
            if np.any(slice_mask):
                area = np.sum(slice_mask)
                slices_with_component.append(z)
                slice_areas[z] = area
        
        if not slices_with_component:
            continue
            
        # Find the slice with the largest area for this component
        best_slice = max(slice_areas, key=slice_areas.get)
        best_slice_area = slice_areas[best_slice]
        
        components_info.append({
            'component_id': comp_id,
            'total_volume': total_volume,
            'slices_with_component': slices_with_component,
            'slice_areas': slice_areas,
            'best_slice': best_slice,
            'best_slice_area': best_slice_area,
            'num_slices': len(slices_with_component),
            'component_3d_mask': component_3d_mask
        })
        
        print(f"Component {comp_id}: {len(slices_with_component)} slices, "
              f"best slice {best_slice} (area: {best_slice_area})")
    
    return components_info, num_components


class Prostate3DInferenceDataset(Dataset):
    def __init__(self, data_path, image_size=1024, min_component_area=10):
        """
        Dataset for inference that extracts 3D connected components and assigns 
        each component its best slice (largest area) as the prompt slice.
        
        Args:
            data_path: Path to the data directory with npz files
            image_size: Size to resize images to
            min_component_area: Minimum area (pixels) a component must have in its best slice
        """
        self.data_path = data_path
        self.img_size = image_size
        self.min_component_area = min_component_area
        
        # Get all npz files
        all_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        all_files.sort()
        
        print(f"Found {len(all_files)} files for inference")
        self.data_files_original = all_files
        
        # Process each file to extract 3D connected components
        self.enhanced_data_list = self._process_files_for_3d_components()
        print(f"Created inference dataset with {len(self.enhanced_data_list)} samples")
        
    def _process_files_for_3d_components(self):
        """
        Process all files to extract 3D connected components and create enhanced data list.
        Each component gets assigned its best slice as the prompt slice.
        """
        enhanced_data_list = []

        print("Processing files to extract 3D connected components...")
        for file_idx, filename in enumerate(tqdm(self.data_files_original, desc="Extracting 3D components")):
            npz_path = os.path.join(self.data_path, filename)
            
            # Get corresponding pkl file
            base_name = os.path.splitext(filename)[0]
            pkl_file = f"{base_name}.pkl"
            pkl_path = os.path.join(self.data_path, pkl_file)
            
            try:
                # Load data
                npz_data = np.load(npz_path)
                volume_seg = npz_data['seg']

                # Ensure correct dimensions for segmentation
                if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
                    volume_seg = volume_seg[0]  # Convert to Slice x H x W

                # Find 3D connected components and their best slices
                components_info, total_components = find_3d_connected_components_and_best_slices(volume_seg)

                # Check if metadata exists
                metadata = None
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, 'rb') as f:
                            metadata = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading metadata from {pkl_path}: {e}")

                if total_components == 0:
                    # No components found, still include the file for inference
                    # with a dummy sample
                    enhanced_data_list.append({
                        'filename': filename,
                        'component_id': 0,
                        'slice_idx': 0,  # For backward compatibility
                        'prompt_slice': 0,
                        'best_slice_area': 0,
                        'component_area': 0,  # For backward compatibility
                        'total_volume': 0,
                        'num_slices': 0,
                        'slices_with_component': [],
                        'total_components_in_file': 0,
                        'components_in_slice': 0,  # For backward compatibility
                        'output_suffix': '',
                        'metadata_path': pkl_path if metadata else None,
                        'component_3d_mask': None
                    })
                else:
                    # Process each 3D component
                    valid_components = 0
                    for comp_info in components_info:
                        # Only include components that meet the minimum area requirement in their best slice
                        if comp_info['best_slice_area'] >= self.min_component_area:
                            # Create a suffix for the output file
                            # Format: _comp{comp_id-1} if multiple components
                            if total_components > 1:
                                suffix = f"_comp{comp_info['component_id']-1}"
                            else:
                                suffix = ""
                            
                            enhanced_data_list.append({
                                'filename': filename,
                                'component_id': comp_info['component_id'],
                                'slice_idx': comp_info['best_slice'],  # For backward compatibility
                                'prompt_slice': comp_info['best_slice'],
                                'best_slice_area': comp_info['best_slice_area'],
                                'component_area': comp_info['best_slice_area'],  # For backward compatibility
                                'total_volume': comp_info['total_volume'],
                                'num_slices': comp_info['num_slices'],
                                'slices_with_component': comp_info['slices_with_component'],
                                'total_components_in_file': total_components,
                                'components_in_slice': 1,  # For backward compatibility - each 3D component is treated as 1
                                'output_suffix': suffix,
                                'metadata_path': pkl_path if metadata else None,
                                'component_3d_mask': comp_info['component_3d_mask']
                            })
                            valid_components += 1
                    
                    if valid_components == 0:
                        # No valid components found, add a dummy sample
                        enhanced_data_list.append({
                            'filename': filename,
                            'component_id': 0,
                            'slice_idx': 0,  # For backward compatibility
                            'prompt_slice': 0,
                            'best_slice_area': 0,
                            'component_area': 0,  # For backward compatibility
                            'total_volume': 0,
                            'num_slices': 0,
                            'slices_with_component': [],
                            'total_components_in_file': 0,
                            'components_in_slice': 0,  # For backward compatibility
                            'output_suffix': '',
                            'metadata_path': pkl_path if metadata else None,
                            'component_3d_mask': None
                        })
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        return enhanced_data_list
        
    def __len__(self):
        return len(self.enhanced_data_list)
        
    def __getitem__(self, index):
        """Get a 3D volume for inference with a specific 3D component and its best prompt slice"""
        # Get the enhanced data entry
        data_entry = self.enhanced_data_list[index]
        filename = data_entry['filename']
        component_id = data_entry['component_id']
        prompt_slice_idx = data_entry['prompt_slice']
        output_suffix = data_entry['output_suffix']
        metadata_path = data_entry['metadata_path']
        component_3d_mask = data_entry['component_3d_mask']

        # Load npz file
        npz_path = os.path.join(self.data_path, filename)
        npz_data = np.load(npz_path)

        # Extract data and segmentation
        volume_data = npz_data['data']  # 5-channel data
        volume_seg = npz_data['seg']  # Segmentation masks
        
        # Get original dimensions before resizing
        original_shape = volume_data.shape[1:]  # [slices, height, width]

        # Process gland masks and bounding boxes
        combined_gland_mask, gland_bbox_mask, gland_bboxes = process_gland_masks_and_generate_boxes(
            volume_data, dilation_pixels=15
        )

        # Use only first 3 channels (as RGB)
        volume_data = volume_data[:3]  # Now 3 x Slice x H x W
        volume_data_original = volume_data.copy()  # Save original data for saving
        volume_data = normalize_channels(volume_data)

        # Ensure correct dimensions for segmentation
        if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
            volume_seg = volume_seg[0]  # Convert to Slice x H x W

        # Extract the specific 3D component
        component_volume_seg = np.zeros_like(volume_seg)
        
        if component_3d_mask is not None and data_entry['total_components_in_file'] > 0:
            # Use the precomputed 3D component mask
            component_volume_seg = component_3d_mask.astype(np.uint8)

        # Get dimensions
        n_channels = volume_data.shape[0]
        z_dim = volume_data.shape[1]
        
        # Use full volume for inference
        video_length = z_dim
        starting_frame = 0

        # Prepare tensors for images and masks
        img_tensor = torch.zeros(video_length, n_channels, self.img_size, self.img_size)
        mask_dict = {}
        prompt_masks = {}

        # Process each slice in our volume
        for z_idx in range(starting_frame, starting_frame + video_length):
            if z_idx >= z_dim:
                break

            # Get slice data and segmentation
            slice_data = np.transpose(volume_data[:, z_idx], (1, 2, 0))  # [H, W, 3]
            slice_seg = component_volume_seg[z_idx]

            # Get unique object IDs in segmentation (non-zero values)
            obj_list = np.unique(slice_seg[slice_seg > 0])
            diff_obj_mask_dict = {}

            # Process each object in the segmentation
            for obj_id in obj_list:
                obj_mask = slice_seg == obj_id

                # Skip if mask is empty
                if not np.any(obj_mask):
                    continue

                # Resize mask using PIL for proper resizing
                obj_mask_pil = Image.fromarray((obj_mask * 255).astype(np.uint8))
                obj_mask_resized = obj_mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)

                # Convert back to binary mask tensor
                obj_mask_np = np.array(obj_mask_resized) > 0
                obj_mask_tensor = torch.tensor(obj_mask_np).unsqueeze(0).int()

                # Store mask
                diff_obj_mask_dict[int(obj_id)] = obj_mask_tensor

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
                
                # If this is the prompt slice, also store in prompt_masks
                if z_idx == prompt_slice_idx:
                    prompt_masks[frame_idx] = diff_obj_mask_dict

        # Load metadata if available
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            except Exception as e:
                print(f"Error loading metadata from {metadata_path}: {e}")

        # Prepare metadata
        image_meta_dict = {
            'filename_or_obj': filename,
            'component_id': component_id,
            'slice_idx': prompt_slice_idx,  # For backward compatibility
            'prompt_slice': prompt_slice_idx,
            'best_slice_area': data_entry['best_slice_area'],
            'component_area': data_entry['best_slice_area'],  # For backward compatibility
            'total_volume': data_entry['total_volume'],
            'num_slices': data_entry['num_slices'],
            'slices_with_component': data_entry['slices_with_component'],
            'total_components_in_file': data_entry['total_components_in_file'],
            'components_in_slice': data_entry['components_in_slice'],  # For backward compatibility
            'output_suffix': output_suffix,
            'original_shape': original_shape,
            'original_data': volume_data_original,  # Store original data for saving as nii.gz
            'metadata': metadata,
            'original_seg': component_volume_seg  # Original segmentation for this component
        }

        # Scale bounding boxes to match image size
        orig_height, orig_width = volume_data.shape[2], volume_data.shape[3]
        scaled_bboxes = []
        
        for bbox in gland_bboxes:
            if bbox is None:
                scaled_bboxes.append(None)
                continue
                
            x_min, y_min, x_max, y_max = bbox
            
            # Calculate scaling factors
            scale_x = self.img_size / orig_width
            scale_y = self.img_size / orig_height
            
            # Scale the coordinates
            x_min_scaled = int(x_min * scale_x)
            y_min_scaled = int(y_min * scale_y)
            x_max_scaled = int(x_max * scale_x)
            y_max_scaled = int(y_max * scale_y)
            
            scaled_bboxes.append((x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled))

        return {
            'image': img_tensor,
            'label': mask_dict,
            'prompt_masks': prompt_masks,
            'image_meta_dict': image_meta_dict,
            'prompt_slice_idx': prompt_slice_idx,
            'gland_bboxes': scaled_bboxes
        }

def analyze_dataset(data_path, min_component_area=10):
    """
    Analyze the dataset and provide detailed statistics about 3D components.
    
    Args:
        data_path: Path to the data directory
        min_component_area: Minimum area for components to be included in their best slice
        
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing dataset structure for 3D components...")
    
    # Get all npz files
    all_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
    all_files.sort()
    
    analysis_results = {
        'total_files': len(all_files),
        'total_samples': 0,
        'files_with_no_components': 0,
        'files_with_components': 0,
        'file_details': []
    }
    
    for filename in tqdm(all_files, desc="Analyzing files for 3D components"):
        npz_path = os.path.join(data_path, filename)
        
        try:
            # Load segmentation data
            npz_data = np.load(npz_path)
            volume_seg = npz_data['seg']
            
            # Ensure correct dimensions
            if volume_seg.ndim == 4 and volume_seg.shape[0] == 1:
                volume_seg = volume_seg[0]
            
            # Analyze 3D components
            components_info, total_components = find_3d_connected_components_and_best_slices(volume_seg)
            
            # Count valid components (meeting area requirement in best slice)
            valid_components = len([comp for comp in components_info 
                                 if comp['best_slice_area'] >= min_component_area])
            
            # Determine samples count for this file
            samples_from_file = max(1, valid_components)  # At least 1 sample per file
            
            file_detail = {
                'filename': filename,
                'total_slices': volume_seg.shape[0],
                'total_3d_components': total_components,
                'valid_3d_components': valid_components,
                'samples_generated': samples_from_file,
                'component_details': []
            }
            
            for comp_info in components_info:
                if comp_info['best_slice_area'] >= min_component_area:
                    file_detail['component_details'].append({
                        'component_id': comp_info['component_id'],
                        'total_volume': comp_info['total_volume'],
                        'num_slices': comp_info['num_slices'],
                        'best_slice': comp_info['best_slice'],
                        'best_slice_area': comp_info['best_slice_area'],
                        'slices_with_component': comp_info['slices_with_component']
                    })
            
            analysis_results['file_details'].append(file_detail)
            analysis_results['total_samples'] += samples_from_file
            
            if valid_components == 0:
                analysis_results['files_with_no_components'] += 1
            else:
                analysis_results['files_with_components'] += 1
                
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            continue
    
    return analysis_results

import sys
def print_analysis_summary(analysis_results, file=sys.stdout):
    """Print a detailed summary of the 3D component dataset analysis."""
    print("\n" + "="*80, file=file)
    print("3D COMPONENT DATASET ANALYSIS SUMMARY", file=file)
    print("="*80, file=file)
    
    print(f"Total files: {analysis_results['total_files']}", file=file)
    print(f"Total samples generated: {analysis_results['total_samples']}", file=file)
    print(f"Files with components: {analysis_results['files_with_components']}", file=file)
    print(f"Files with no components: {analysis_results['files_with_no_components']}", file=file)
    
    print("\n" + "-"*80, file=file)
    print("DETAILED FILE BREAKDOWN", file=file)
    print("-"*80, file=file)
    
    for file_detail in analysis_results['file_details']:
        print(f"\nFile: {file_detail['filename']}", file=file)
        print(f"  Total slices: {file_detail['total_slices']}", file=file)
        print(f"  Total 3D components found: {file_detail['total_3d_components']}", file=file)
        print(f"  Valid 3D components (volume >= min): {file_detail['valid_3d_components']}", file=file)
        print(f"  Samples generated: {file_detail['samples_generated']}", file=file)
        
        if file_detail['component_details']:
            print("  Component breakdown:", file=file)
            for comp_detail in file_detail['component_details']:
                slices_str = ", ".join([str(s) for s in comp_detail['slices_with_component']])
                print(f"    Component {comp_detail['component_id']:2d}: "
                      f"volume={comp_detail['total_volume']:4d}, "
                      f"spans {comp_detail['num_slices']:2d} slices, "
                      f"best_slice={comp_detail['best_slice']:2d} (area={comp_detail['best_slice_area']:3d}), "
                      f"slices=[{slices_str}]", file=file)

if __name__ == "__main__":
    # Example usage and analysis
    data_path = "/data/pct_ids_nvme/users/z005257c/nnUNet_preprocessed/Dataset202_QuadiaSlice/nnUNetPlans_2d"  
    min_component_area = 10
    
    print("Creating dataset with 3D component analysis...")
    dataset = Prostate3DInferenceDataset(
        data_path=data_path,
        image_size=1024,
        min_component_area=min_component_area
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Perform detailed analysis
    print("\nPerforming detailed 3D component analysis...")
    analysis_results = analyze_dataset(data_path, min_component_area)
    
    # Print to console
    print_analysis_summary(analysis_results)
    
    # Save to file
    with open("3d_component_analysis_summary.txt", "w") as f:
        print_analysis_summary(analysis_results, file=f)
    
    print(f"\nAnalysis saved to: 3d_component_analysis_summary.txt")