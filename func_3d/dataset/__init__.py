from .btcv import BTCV
from .amos import AMOS
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .prostate_dataset import Prostate3DDataset
from .prostate_dataset_boxes import Prostate3DDataset as Prostate3DDatasetBoxes
from .prostate_dataset_boxes_isolated import Prostate3DDatasetWithConnectedComponents
from torch.utils.data._utils.collate import default_collate

# def custom_collate_fn(batch):
#     """
#     Custom collate function that handles None values in the batch.
#     Converts None values to empty tensors or lists as appropriate.

#     Args:
#         batch: A list of samples from the dataset

#     Returns:
#         Collated batch that PyTorch DataLoader can process
#     """
#     elem = batch[0]
#     if isinstance(elem, tuple):
#         return tuple(custom_collate_fn(samples) for samples in zip(*batch))

#     elif isinstance(elem, dict):
#         # Process each key in the dictionary
#         result = {}
#         for key in elem:
#             # Special handling for gland_bboxes
#             if key == 'gland_bboxes':
#                 # Convert None values to empty tensors or a fixed default
#                 result[key] = [sample[key] if sample[key] is not None else (0, 0, 1, 1) for sample in batch]
#             else:
#                 try:
#                     result[key] = default_collate([sample[key] for sample in batch])
#                 except TypeError:
#                     # Handle other None values in the dictionary
#                     if all(sample[key] is None for sample in batch):
#                         result[key] = None
#                     else:
#                         # For mixed None/non-None, use a list
#                         result[key] = [sample[key] for sample in batch]
#         return result

#     elif elem is None:
#         return None

#     # Default handling for other types
#     try:
#         return default_collate(batch)
#     except TypeError:
#         return batch

def custom_collate_fn(batch):
    result = {}
    # Process non-nested dictionary keys first
    for key in ['image', 'prompt_slice_idx', 'image_meta_dict', 'gland_bboxes']:
        if all(key in sample for sample in batch):
            result[key] = default_collate([sample[key] for sample in batch])
    
    # Handle nested dictionaries separately
    for key in ['label', 'prompt_masks', 'bbox', 'p_label', 'pt']:
        if all(key in sample for sample in batch):
            # Don't try to collate these - keep them as lists of dictionaries
            result[key] = [sample[key] for sample in batch]
    
    return result

def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'prostate':
        prostate_train_dataset = Prostate3DDataset(args, args.data_path, transform = None, mode = 'Training', prompt=args.prompt, filter_empty=True)
        prostate_val_dataset = Prostate3DDataset(args, args.data_path, transform = None, mode = 'Validation', prompt=args.prompt, filter_empty=True)
        nice_train_loader = DataLoader(prostate_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(prostate_val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    elif args.dataset == 'prostate_box':
        prostate_train_dataset = Prostate3DDatasetBoxes(args, args.data_path, transform=None, mode='Training',
                                                   prompt=args.prompt, filter_empty=True)
        prostate_val_dataset = Prostate3DDatasetBoxes(args, args.data_path, transform=None, mode='Validation',
                                                 prompt=args.prompt, filter_empty=True)
        nice_train_loader = DataLoader(prostate_train_dataset, batch_size=1, shuffle=True, num_workers=8,
                                       pin_memory=True, collate_fn=custom_collate_fn)
        nice_test_loader = DataLoader(prostate_val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                      collate_fn=custom_collate_fn)
    elif args.dataset == 'prostate_box_isolated':
        prostate_train_dataset = Prostate3DDatasetWithConnectedComponents(args, args.data_path, mode='Training', prompt=args.prompt, seed=42, min_component_slices=1)
        prostate_val_dataset = Prostate3DDatasetWithConnectedComponents(args, args.data_path, mode='Validation', prompt=args.prompt, seed=42, min_component_slices=1)
        nice_train_loader = DataLoader(prostate_train_dataset, batch_size=1, shuffle=True, num_workers=8,
                                       pin_memory=True, collate_fn=custom_collate_fn)
        nice_test_loader = DataLoader(prostate_val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                      collate_fn=custom_collate_fn)


    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader