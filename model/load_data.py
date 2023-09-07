import os
from glob import glob
import natsort
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.opticalflow import OpticalFlowProcessor
from lib.utils import rgb2mask, rgb_to_gray_tensor
from lib.extract_patches import random_patches, augment_rectangular
import json
import cv2 as cv
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.functional import uniform_temporal_subsample_repeated
from torchvision.utils import save_image

class SPDataset(Dataset):
    def __init__(self, path, config_data):
        self.path_images = natsort.natsorted(glob(os.path.join(path, 'images', '*.png')))
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.png')))
        self.config_data = config_data
        self._initialize_attributes()
        self._prepare_sequences()
        self._prepare_transform()
        self._prepare_final_images_with_flow()
        

    def __len__(self):
        return len(self.final_images)
                
    
    def _initialize_attributes(self):
        self.use_augmentation = self.config_data["training_parameters"]["use_augmentation"]
        self.use_patches = self.config_data["training_parameters"]["use_patches"]
        self.use_optical_flow = self.config_data["training_parameters"]["use_optical_flow"]
        self.optical_flow_function_name = self.config_data["data_preprocessing"]["optical_flow_function_name"]
        self.seq_len = self.config_data["training_parameters"]["sequence_length"]


    def _prepare_sequences(self):
        split_images = [self.path_images[x:x+self.seq_len] for x in range(0, len(self.path_images), self.seq_len)]
        split_masks = [self.path_masks[x:x+self.seq_len] for x in range(0, len(self.path_images), self.seq_len)]
        self.sequences = [{'images': images_sublist, 'masks': masks_sublist, 'name': i}
                            for i, (images_sublist, masks_sublist) in enumerate(zip(split_images, split_masks))]

    def _prepare_transform(self):
        if self.use_augmentation:
             options = [
                    A.Rotate(limit=35, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
        else:
            options = [
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]

        self.transform = A.Compose(
                options,
                additional_targets={

                        'image1': 'image',
                        'image2': 'image',
                        'image3': 'image',
                        'image4': 'image',
                        'image5': 'image',
                        'image6': 'image',
                        'image7': 'image',
                        'image8': 'image',
                        'image9': 'image',


                        'mask1': 'mask',
                        'mask2': 'mask',
                        'mask3': 'mask',
                        'mask4': 'mask',
                        'mask5': 'mask',
                        'mask6': 'mask',
                        'mask7': 'mask',
                        'mask8': 'mask',
                        'mask9': 'mask'}
            )
        
    
    def _load_images_and_masks(self, idx):
        imgs = [np.array(Image.open(img)) for img in self.sequences[idx]['images']]
        targets = [rgb2mask(np.array(Image.open(mask))) for mask in self.sequences[idx]['masks']]
        return imgs, targets

    def _apply_augmentations(self, imgs, targets):
        if self.transform:
            augmentations = self.transform(image=imgs[0], **{f'image{i}': img for i, img in enumerate(imgs[1:], start=1)},
                                           mask=targets[0], **{f'mask{i}': mask for i, mask in enumerate(targets[1:], start=1)})
         

            imgs = [augmentations[f'image{i}'] for i in range(1,self.seq_len)]
            targets = [augmentations[f'mask{i}'] for i in range(1,self.seq_len)]
            imgs.insert(0, augmentations['image'])
            targets.insert(0, augmentations['mask'])
        return imgs, targets
    # returns a whole sequence

    def _prepare_final_images_with_flow(self):
        self.final_images = []
        self.final_masks = []

        for idx, seq in enumerate(self.sequences):
            imgs, targets = self._load_images_and_masks(idx)
            imgs, targets = self._apply_augmentations(imgs, targets)
            
            if self.use_optical_flow:
                opticalflow = OpticalFlowProcessor(self.config_data)
                self.optical_flow_function = opticalflow.return_chosen_method(self.optical_flow_function_name)
                imgs, targets = self.optical_flow_function(imgs, targets) #4,512,512
                
            if self.use_patches:
                imgs, targets = self.prepare_patches(imgs, targets)

            for i in range(len(imgs)):
                self.final_images.append(imgs[i])
                self.final_masks.append(targets[i].long())

    def __getitem__(self, idx):
        
        sample = {'image':self.final_images[idx], 'mask':self.final_masks[idx]}
    
        return sample

    
    def prepare_patches(self, imgs, targets):
        n = self.config_data["patches"]["number_of_patches_per_image"] 
        shape_to_label = self.config_data["models_settings"]["label_to_value"]
        label_to_shape = {v:k for k,v in shape_to_label.items()}
        d_pixels = {k:0 for k in shape_to_label.keys()}
        d_images = {k:0 for k in shape_to_label.keys()}
        feature_labels = set([v for k,v in shape_to_label.items() if k!='_background_'])

        patches_img = []
        patches_mask = []

        for img, mask in zip(imgs, targets):

            patches, patch_masks = random_patches(img, mask, n=n, patch_h=self.config_data["patches"]["patch_height"], patch_w=self.config_data["patches"]["patch_width"])
            for patch, patch_mask in zip(patches,patch_masks):
                # consider only patch containing feature_labels
                if len(set(np.unique(patch_mask)).intersection(feature_labels))>0:  
                    # data.append((patch, patch_mask))
                    patch = np.transpose(patch, (2,0,1))
                    patches_img.append(patch)
                    patches_mask.append(patch_mask)
                    
                    count = Counter(patch_mask.flatten().tolist())
                    for label, num in count.most_common():
                        d_pixels[label_to_shape[label]] += num
                        d_images[label_to_shape[label]] += 1
                        
        print('pixel per class = ', d_pixels)
        print('images per class = ', d_images)

        return patches_img, patches_mask

class CoreDataset(Dataset):
    
    def __init__(self, path, transform=None):
        self.path_images = natsort.natsorted(glob(os.path.join(path, 'images', '*.png')))
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.png')))
        self.transform = transform
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.path_images[idx])
        mask = Image.open(self.path_masks[idx])
        
        sample = {'image':image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
       
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels
        mask = rgb2mask(np.array(mask))
        sample = {'image': torch.from_numpy(image).permute(2,0,1).float(),
                  'mask': torch.from_numpy(mask).long(), 
                 }
        
        return sample
    
def make_datasets(path, val_ratio, config_data):
    if config_data["training_parameters"]["spdataset"]:
        dataset = SPDataset(path, config_data)
    else:
        dataset = CoreDataset(path, transform = transforms.Compose([ToTensor()]))
    val_len = int(val_ratio*len(dataset))
    lengths = [len(dataset)-val_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lengths)
    
    return train_dataset, val_dataset


def make_dataloaders(path, val_ratio, config_data, params):
    train_dataset, val_dataset = make_datasets(path, val_ratio, config_data)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    
    return train_loader, val_loader