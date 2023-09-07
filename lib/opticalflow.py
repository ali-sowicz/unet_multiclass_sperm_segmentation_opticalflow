

import numpy as np
import torch
from typing import Tuple, List
from lib.utils import rgb_to_gray_tensor
import cv2 as cv
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.functional import uniform_temporal_subsample_repeated

class OpticalFlowProcessor:
    def __init__(self, config_data):
        self.config_data = config_data
        self.params= self.config_data['slowfast']
        self.original_img_size= config_data['data_preprocessing']['original_image_size']
        self._prepare_slowfast()

    def _prepare_slowfast(self):
        self.slow_fast = create_slowfast(
            slowfast_channel_reduction_ratio= tuple(self.params['slowfast_channel_reduction_ratio']),
            slowfast_conv_channel_fusion_ratio= self.params['slowfast_conv_channel_fusion_ratio'],
            slowfast_fusion_conv_kernel_size=tuple(self.params['slowfast_fusion_conv_kernel_size']), 
            slowfast_fusion_conv_stride = tuple(self.params['slowfast_fusion_conv_stride']), 
            input_channels=(self.params['input_channels'],) * 2,
            model_depth=self.params['model_depth'],
            model_num_class=self.params['model_num_class'],
            dropout_rate=self.params['dropout_rate'],
            norm=torch.nn.BatchNorm3d,
            activation=torch.nn.Sigmoid,
            head_pool_kernel_sizes = (tuple(self.params['head_pool_kernel_size0']), tuple(self.params['head_pool_kernel_size1'])),
        ) 

    def _get_inputs(
        self,
        shapes: List[torch.tensor],
        frame_ratios: Tuple[int] = (4, 1),
        clip_length: int = 8,
        ) -> torch.tensor:
            """
            Generate different tensors as test case inputs by subsampling temporal dimensions.

            Args:
                shapes (List[torch.tensor]): List of tensors representing input shapes.
                frame_ratios (Tuple[int]): Frame ratios for temporal subsampling.
                clip_length (int): Length of video clips.

            Yields:
                torch.tensor: Tensor as a test case input.
            """
            # Prepare random inputs as test cases.
            # shapes = ((1, channel, clip_length, crop_size, crop_size),)
            for shape in shapes:
                shape = shape.reshape((-1,) + (clip_length, 3, self.original_img_size, self.original_img_size))
                shape = np.transpose(shape, (0, 2, 1, 3, 4))
                
                yield uniform_temporal_subsample_repeated(
                    shape, frame_ratios=frame_ratios, temporal_dim=2
                )

    def slowfast_images(self, imgs, targets):
        """
        Calculate optical flow and append it to image frames using a SlowFast model.

        Args:
            imgs (list of torch.Tensor): A list of input image frames, each as a torch.Tensor.
            targets (list of torch.Tensor): A list of target masks, each as a torch.Tensor.
                                           Each target mask corresponds to an image frame in 'imgs'.

        Returns:
            tuple: A tuple containing:
                - imgs_with_flow (list of torch.Tensor): A list of image frames with appended optical flow,
                  where each element is a torch.Tensor.
                - targets (list of torch.Tensor): The unmodified list of target masks.
        """
        
        imgs_frames_stacked = [torch.stack(imgs[:self.params['frames_length']])]
        for tensor in self._get_inputs(
            shapes=imgs_frames_stacked, frame_ratios=(self.params['slow_pathway_size'],self.params['fast_pathway_size']), clip_length=self.params['frames_length']
            ):
                with torch.no_grad():
                    flow = self.slow_fast(tensor)
                    
        
        flow = flow.unsqueeze(2) #1,512,1
        flow = flow.repeat(1,1,self.original_img_size).reshape(1,self.original_img_size,self.original_img_size)
        for i, im in enumerate(imgs):
            imgs[i] = torch.cat([im, flow])

        return imgs, targets
    
    def slowfast_images_iterative(self, imgs, targets):
        """
        Perform an iterative process to calculate optical flow and append it to image frames using a SlowFast model.

        Args:
            imgs (list of torch.Tensor): A list of input image frames, each as a torch.Tensor.
            targets (list of torch.Tensor): A list of target masks, each as a torch.Tensor.
                                           Each target mask corresponds to an image frame in 'imgs'.

        Returns:
            tuple: A tuple containing:
                - imgs_with_flow (list of torch.Tensor): A list of image frames with appended optical flow,
                  where each element is a torch.Tensor.
                - targets (list of torch.Tensor): The unmodified list of target masks.
        """
        imgs_with_flow = imgs.copy()
        for i, img in enumerate(imgs):
            if i <= 8:
               
                frame_1st = imgs[i] #3,512,512
                frame_2st = imgs[i+1]
                two_frames = [torch.stack([frame_1st,frame_2st])]

                for tensor in self._get_inputs(
                two_frames, frame_ratios=(2,1), clip_length=2
            ):
                    with torch.no_grad():
                        flow = self.slow_fast(tensor)
                        flow = flow.unsqueeze(2) #1,512,1
                        flow = flow.repeat(1,1,self.original_img_size).reshape(1,self.original_img_size,self.original_img_size)

                if i == 0:
                    imgs_with_flow[0] =  torch.cat([imgs[0].clone().detach(), flow])
                    imgs_with_flow[1] =  torch.cat([imgs[1].clone().detach(), flow])

                else:
                    imgs_with_flow[i+1] =  torch.cat([imgs[i+1].clone().detach(), flow])

        return imgs_with_flow, targets
    
    def TVL1_iterative(self, imgs, targets):
        """
        Perform an iterative process to calculate optical flow and append it to image frames.

        This function takes a sequence of image frames and corresponding target masks. It calculates
        optical flow between consecutive frames using the TV-L1 algorithm and appends the flow
        information to the image frames. This iterative process is useful for providing temporal
        information to neural networks during training.

        Args:
            imgs (list of torch.Tensor): A list of input image frames, each as a torch.Tensor.
            targets (list of torch.Tensor): A list of target masks, each as a torch.Tensor.
                                           Each target mask corresponds to an image frame in 'imgs'.

        Returns:
            tuple: A tuple containing:
                - imgs_with_flow (list of torch.Tensor): A list of image frames with appended optical flow,
                  where each element is a torch.Tensor.
                - targets (list of torch.Tensor): The unmodified list of target masks.
        """
        imgs_with_flow = imgs.copy()

        for i, img in enumerate(imgs):
            if i <= 8:
                frame_1st = imgs[i] #3,512,512
                frame_2st = imgs[i+1]


                flow = self.TVL1(frame_1st,frame_2st)
                
                if i == 0:
                    imgs_with_flow[0] =  torch.cat([imgs[0].clone().detach(), flow])
                    imgs_with_flow[1] =  torch.cat([imgs[1].clone().detach(), flow])
                else:
                    imgs_with_flow[i+1] =  torch.cat([imgs[i+1].clone().detach(), flow])

        return imgs_with_flow, targets
    
    def TVL1(self, frame1, frame2):
        """
        Calculate dense optical flow using the TV-L1 algorithm.

        Args:
            frame1 (torch.Tensor): First frame as a torch tensor with shape (3, 512, 512).
            frame2 (torch.Tensor): Second frame as a torch tensor with shape (3, 512, 512).

        Returns:
            torch.Tensor: Dense optical flow as a torch tensor with shape (2, 512, 512),
                         representing horizontal and vertical flow components.
        """
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
    
    def stack_and_get_middle_masks(self, imgs, targets):
        """
        Stack consecutive image frames and extract middle masks for object detection.

        This function takes a sequence of image frames and corresponding target masks.
        It stacks a specified number of consecutive frames and extracts the middle mask
        corresponding to each stack. This is useful for training convolutional neural networks
        (CNNs) to predict object positions in the mid frame of a sequence.

        Args:
            imgs (list of torch.Tensor): A list of input image frames, each as a torch.Tensor.
                                        The list should contain consecutive frames in the sequence.
            targets (list of torch.Tensor): A list of target masks, each as a torch.Tensor.
                                           Each target mask corresponds to an image frame in 'imgs'.

        Returns:
            tuple: A tuple containing:
                - imgs_stacked (list of torch.Tensor): A list of stacked image frames, where each
                  element is a stack of consecutive frames as a torch.Tensor.
                - masks_middle (list of torch.Tensor): A list of middle masks corresponding to each
                  stacked frame in 'imgs_stacked' as torch.Tensors.
        """
        # n_imgs is the number of frames in a stack
        n_imgs = self.config_data["data_preprocessing"]["number_of_stacked_imgs"]

        imgs_stacked = []

        # List to store the middle masks corresponding to each stack
        masks_middle = []
        for i in range (len(imgs)-(n_imgs-1)): 
            # Extract the consecutive image frames for the stack
            frames = imgs[i:i+n_imgs] 
            gray_frames = [rgb_to_gray_tensor(item) for item in frames]
            
            stacked_frames = torch.stack(gray_frames)
            
            # Get the middle mask corresponding to the current stack
            middle_mask = targets[i + n_imgs // 2]
            
            imgs_stacked.append(stacked_frames)
            masks_middle.append(middle_mask)

        return imgs_stacked, masks_middle
    
    def return_chosen_method(self, method_name):
        method = getattr(self, method_name, None)
        return method