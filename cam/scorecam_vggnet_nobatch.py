# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 01:59:04 2021

@author: maeng
"""

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        #conv_output, model_output = extractor.forward_pass(input_image)

        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.zeros((len(target_class), target.shape[1], target.shape[2]), dtype=np.float32)
        #cam = np.zeros((len(target_class), input_image.shape[2], input_image.shape[3]), dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            prop = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            #prop = F.softmax(extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            
            for idx, w in enumerate(prop):
            
                cam[idx, :, :] += w.detach().cpu().numpy() * target[i, :, :].detach().cpu().numpy()
                #cam[idx, :, :] += w.detach().cpu().numpy() * saliency_map.squeeze().detach().cpu().numpy()
        return cam
