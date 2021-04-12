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
        self.extractor = CamExtractor(self.model, target_layer)
        #extractor = CamExtractor(model, TARGET_LAYER)
        
    def generate_cam(self, input_image, target_class, batch_size):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        #conv_output, model_output = extractor.forward_pass(img)

        target = conv_output[0].detach()

        cam = np.zeros((len(target_class), target.shape[1], target.shape[2]), dtype = np.float32)
        #cam = np.zeros((len(label), target.shape[1], target.shape[2]), dtype = np.float32)

        target = torch.utils.data.DataLoader(target, batch_size = batch_size)

        for i in target:
            #break
            saliency_map = F.interpolate(i.unsqueeze(0), size = (224, 224), mode = 'bilinear', align_corners=False).squeeze()
            max_val = saliency_map.view(saliency_map.shape[0], -1).max(1)[0].unsqueeze(-1).unsqueeze(-1)
            min_val = saliency_map.view(saliency_map.shape[0], -1).min(1)[0].unsqueeze(-1).unsqueeze(-1)

            saliency_map = (saliency_map - min_val) / (max_val - min_val)
            saliency_map = saliency_map.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
            
            idx = [j for j in range(len(saliency_map)) if torch.isnan(saliency_map[j, :, :].sum())]
            
            prop = F.softmax(self.extractor.forward_pass(input_image * saliency_map)[1],dim=1)[:, target_class].detach()
            #prop = F.softmax(extractor.forward_pass(img * saliency_map)[1], dim=1)[:, target_class].detach()
            prop[idx, :] = torch.zeros(1, len(target_class)).to("cuda")
            
            for k, l in zip(prop, i):
                cam += (k.unsqueeze(-1).unsqueeze(-1) * l.repeat(len(target_class), 1, 1)).cpu().numpy()
            
        return cam
            



