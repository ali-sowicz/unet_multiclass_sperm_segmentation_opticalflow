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
from lib.utils import rgb2mask
from lib.extract_patches import random_patches, augment_rectangular
import json
import cv2 as cv
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.functional import uniform_temporal_subsample_repeated
from torchvision.utils import save_image
# PATH_PARAMETERS = './config.json'

class SPDataset(Dataset):
    def __init__(self, path, config_data):
        self.path_images = natsort.natsorted(glob(os.path.join(path, 'images', '*.png')))
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.png')))
        self.params= config_data['data_preprocessing']
        self.models_settings = config_data['models_settings']
        self.training_parameters = config_data['training_parameters']
        self._initialize_attributes()
        self._prepare_sequences()
        self._prepare_transform()
        self._prepare_slowfast()
        self._prepare_final_images_with_flow()
        

    def __len__(self):
        # return len(self.sequences)
        return len(self.final_images)
                
    
    def _initialize_attributes(self):
        # c = globals()
        self.use_augmentation = self.params["use_augmentation"]
        self.use_patches = self.params["use_patches"]
        self.use_optical_flow = self.params["use_optical_flow"]
        self.optical_flow_function_name = self.params['optical_flow_function_name']
        self.frames_length = self.params['frames_length']
        self.slow_pathway_size = self.params['slow_pathway_size']
        self.fast_pathway_size = self.params['fast_pathway_size']
        self.slowfast_channel_reduction_ratio = tuple(self.params['slowfast_channel_reduction_ratio'])
        self.slowfast_fusion_conv_stride = tuple(self.params['slowfast_fusion_conv_stride'])
        self.slowfast_fusion_conv_kernel_size = tuple(self.params['slowfast_fusion_conv_kernel_size'])
        self.head_pool_kernel_size0 = tuple(self.params['head_pool_kernel_size0'])
        self.head_pool_kernel_size1 = tuple(self.params['head_pool_kernel_size1'])
        self.slowfast_conv_channel_fusion_ratio = self.params['slowfast_conv_channel_fusion_ratio']
        self.input_channels = self.params['input_channels']
        self.model_depth = self.params['model_depth']
        self.model_num_class = self.params['model_num_class']
        self.dropout_rate = self.params['dropout_rate']
        self.n_imgs = self.params['n_imgs']
        self.norm = torch.nn.BatchNorm3d
        self.activation = torch.nn.Sigmoid
    

    def _prepare_sequences(self):
        split_images = [self.path_images[x:x+10] for x in range(0, len(self.path_images), 10)]
        split_masks = [self.path_masks[x:x+10] for x in range(0, len(self.path_images), 10)]
        self.sequences = [{'images': images_sublist, 'masks': masks_sublist, 'name': i}
                            for i, (images_sublist, masks_sublist) in enumerate(zip(split_images, split_masks))]

    def _prepare_transform(self):
         self.transform = A.Compose(
                [
                    A.Rotate(limit=35, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ],
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
        
    
    def _prepare_slowfast(self):
        self.slow_fast = create_slowfast(
            slowfast_channel_reduction_ratio= self.slowfast_channel_reduction_ratio,
            slowfast_conv_channel_fusion_ratio= self.slowfast_conv_channel_fusion_ratio,
            slowfast_fusion_conv_kernel_size=self.slowfast_fusion_conv_kernel_size, 
            slowfast_fusion_conv_stride = self.slowfast_fusion_conv_stride, 
            input_channels=(self.input_channels,) * 2,
            model_depth=self.model_depth,
            model_num_class=self.model_num_class,
            dropout_rate=self.dropout_rate,
            norm=self.norm,
            activation=self.activation,
            head_pool_kernel_sizes = (self.head_pool_kernel_size0, self.head_pool_kernel_size1),
        ) 

    def _get_inputs(
        self,
        shapes: List[torch.tensor],
        frame_ratios: Tuple[int] = (4, 1),
        clip_length: int = 8,
        ) -> torch.tensor:
            """
            Provide different tensors as test cases.

            Yield:
                (torch.tensor): tensor as test case input.
            """
            # Prepare random inputs as test cases.
            # shapes = ((1, channel, clip_length, crop_size, crop_size),)
            for shape in shapes:
                
                shape = shape.reshape((-1,) + (clip_length, 3, 512, 512))
                shape = np.transpose(shape, (0, 2, 1, 3, 4))
                
                yield uniform_temporal_subsample_repeated(
                    shape, frame_ratios=frame_ratios, temporal_dim=2
                )

    def _load_images_and_masks(self, idx):
        imgs = [np.array(Image.open(img)) for img in self.sequences[idx]['images']]
        targets = [rgb2mask(np.array(Image.open(mask))) for mask in self.sequences[idx]['masks']]
        return imgs, targets

    def _apply_augmentations(self, imgs, targets):
        if self.transform:
            augmentations = self.transform(image=imgs[0], **{f'image{i}': img for i, img in enumerate(imgs[1:], start=1)},
                                           mask=targets[0], **{f'mask{i}': mask for i, mask in enumerate(targets[1:], start=1)})
         

            imgs = [augmentations[f'image{i}'] for i in range(1,10)]
            targets = [augmentations[f'mask{i}'] for i in range(1,10)]
            imgs.insert(0, augmentations['image'])
            targets.insert(0, augmentations['mask'])
        return imgs, targets
    # returns a whole sequence

    def _prepare_final_images_with_flow(self):
        self.final_images = []
        self.final_masks = []

        for idx, seq in enumerate(self.sequences):
            imgs, targets = self._load_images_and_masks(idx)

            if self.use_augmentation:
                imgs, targets = self._apply_augmentations(imgs, targets)
            # self.save_augumented_imgs_check(imgs,targets,idx)
            
            if self.use_optical_flow:
                self.optical_flow_function = getattr(self, self.optical_flow_function_name, None)
                imgs, targets = self.optical_flow_function(imgs, targets) #4,512,512
                
            if self.use_patches:
                imgs, targets = self.prepare_patches(imgs, targets)
                # imgs, targets = augment_rectangular(imgs, targets)
                # self.save_augumented_imgs_check(imgs,targets,idx)
            if self.training_parameters['train_with_sequences']:
                img_sequence = [img for img in imgs]
                mask_sequence = [target.long() for target in targets]
                
                self.final_images.append(img_sequence)
                self.final_masks.append(mask_sequence)
                # for i in range(len(imgs)):
                #     self.final_images[idx].append(imgs[i])
                #     self.final_masks[idx].append(targets[i].long())
            else:
                for i in range(len(imgs)):
                    self.final_images.append(imgs[i])
                    self.final_masks.append(targets[i].long())

    def __getitem__(self, idx):
        
        sample = {'image':self.final_images[idx], 'mask':self.final_masks[idx]}
    
        return sample

    
    def slowfast_images(self, imgs, targets):
        
        # x_tensor = torch.tensor
        imgs_frames_stacked = [torch.stack(imgs[:self.frames_length])]
        for tensor in self._get_inputs(
            shapes=imgs_frames_stacked, frame_ratios=(self.slow_pathway_size,self.fast_pathway_size), clip_length=self.frames_length
            ):
                with torch.no_grad():
                    flow = self.slow_fast(tensor)
                    # x_tensor = x
        
        flow = flow.unsqueeze(2) #1,512,1
        flow = flow.repeat(1,1,512).reshape(1,512,512)
        # save_image(x_3D, "seq_"+str(idx)+".png")
        for i, im in enumerate(imgs):
            imgs[i] = torch.cat([im, flow])

        return imgs, targets
    
    def slowfast_images_iterative(self, imgs, targets):
        imgs_with_flow = imgs.copy()
        for i, img in enumerate(imgs):
            if i <= 8:
                # frame_1st = np.transpose(imgs[i], (2,0,1)).clone().detach()
                # frame_2st = np.transpose(imgs[i+1], (2,0,1)).clone().detach()
                frame_1st = imgs[i] #3,512,512
                frame_2st = imgs[i+1]
                # frame_2st = torch.tensor(np.transpose(imgs[i+1], (2,0,1))).clone().detach()
                two_frames = [torch.stack([frame_1st,frame_2st])]

                for tensor in self._get_inputs(
                two_frames, frame_ratios=(2,1), clip_length=2
            ):
                    with torch.no_grad():
                        flow = self.slow_fast(tensor)
                        flow = flow.unsqueeze(2) #1,512,1
                        flow = flow.repeat(1,1,512).reshape(1,512,512)

                if i == 0:
                    imgs_with_flow[0] =  torch.cat([imgs[0].clone().detach(), flow])
                    imgs_with_flow[1] =  torch.cat([imgs[1].clone().detach(), flow])
                    # imgs_with_flow[0] =  torch.cat([torch.tensor(imgs[0]), flow])
                    # imgs_with_flow[1] =  torch.cat([torch.tensor(imgs[1]), flow])
                    # imgs[0] =  torch.cat([torch.tensor(np.transpose(imgs[0], (2,0,1))), flow])
                    # imgs[1] =  torch.cat([torch.tensor(np.transpose(imgs[1], (2,0,1))), flow])
                else:
                    imgs_with_flow[i+1] =  torch.cat([imgs[i+1].clone().detach(), flow])
                    # imgs[i+1] =  torch.cat([torch.tensor(np.transpose(imgs[i+1], (2,0,1))), flow])

        return imgs_with_flow, targets
    
    def TVL1_iterative(self, imgs, targets):
        imgs_with_flow = imgs.copy()

        for i, img in enumerate(imgs):
            if i <= 8:
                frame_1st = imgs[i] #3,512,512
                frame_2st = imgs[i+1]
                # frame_1st = torch.tensor(np.transpose(imgs[i], (2,0,1)))
                # frame_2st = torch.tensor(np.transpose(imgs[i+1], (2,0,1)))

                flow = self.TVL1(frame_1st,frame_2st)
                
                if i == 0:
                    imgs_with_flow[0] =  torch.cat([imgs[0].clone().detach(), flow])
                    imgs_with_flow[1] =  torch.cat([imgs[1].clone().detach(), flow])
                    # imgs_with_flow[1] =  torch.cat([torch.tensor(np.transpose(imgs[1], (2,0,1))), flow])
                else:
                    imgs_with_flow[i+1] =  torch.cat([imgs[i+1].clone().detach(), flow])
                    # imgs_with_flow[i+1] =  torch.cat([torch.tensor(np.transpose(imgs[i+1], (2,0,1))), flow])

        return imgs_with_flow, targets
    
    def TVL1(self, frame1, frame2):
        optical_flow = cv.optflow.DualTVL1OpticalFlow_create(0.25, 0.3, 0.3, 1, 5, 0.01, 30, 10, 0.8, 0.0, 5, False)
        
        frame1 = frame1.numpy()
        frame2 = frame2.numpy()
        frame1 = np.transpose(frame1, (1,2,0)) #512,512,3
        frame2 = np.transpose(frame2, (1,2,0))

        hsvx = np.zeros_like(frame1)
        hsvx[...,1] = 10
        hsvx[...,0] =10
        hsvy = np.zeros_like(frame1)
        hsvy[...,1] = 10
        hsvy[...,0]=10

        hsv_mask = np.zeros_like(frame1)
        # Make image saturation to a maximum value
        hsv_mask[..., 1] = 255

        alpha = 2 # Simple contrast control
        beta = 0    # Simple brightness control
        frame1 = cv.convertScaleAbs(frame1, alpha=alpha, beta=beta)
        frame1 = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

        frame2 = cv.convertScaleAbs(frame2, alpha=alpha, beta=beta)
        frame2 = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = optical_flow.calc(frame1, frame2, None)
        flow = torch.from_numpy(flow) #512,512,2
        flow = np.transpose(flow, (2,0,1))

        return flow
    
    def channel_stack_with_middle_mask(self, imgs, targets):
        # n_imgs is the number of frames in a stack
        n_imgs = self.n_imgs

        # List to store the stacked image frames
        imgs_stacked = []

        # List to store the middle masks corresponding to each stack
        masks_middle = []
        for i in range (len(imgs)-(n_imgs-1)): #4 or 2 if n 5 or 3
            # Extract the consecutive image frames for the stack
            frames = imgs[i:i+n_imgs] 
            gray_frames = [self.rgb_to_gray_tensor(item) for item in frames]
            
            stacked_frames = torch.stack(gray_frames)
            
            # Get the middle mask corresponding to the current stack
            middle_mask = targets[i + n_imgs // 2]
            
            imgs_stacked.append(stacked_frames)
            masks_middle.append(middle_mask)

        return imgs_stacked, masks_middle

    def rgb_to_gray_tensor(self,rgb_tensor):
        gray_tensor = 0.2989 * rgb_tensor[0] + 0.5870 * rgb_tensor[1] + 0.1140 * rgb_tensor[2]
        return gray_tensor
        
    def prepare_patches(self, imgs, targets):
        n = 20 #number of patches
        shape_to_label = self.models_settings['label_to_value']
        label_to_shape = {v:k for k,v in shape_to_label.items()}
        d_pixels = {k:0 for k in shape_to_label.keys()}
        d_images = {k:0 for k in shape_to_label.keys()}
        feature_labels = set([v for k,v in shape_to_label.items() if k!='_background_'])

        # data = []
        patches_img = []
        patches_mask = []

        for img, mask in zip(imgs, targets):
            # mask = rgb2mask(mask)
            # mask to array img to array
            patches, patch_masks = random_patches(img, mask, n=n, patch_h=self.models_settings['patch_height'], patch_w=self.models_settings['patch_width'])
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
    
# add image normalization transform at some point
   
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