# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn

class SingleResolutionNeck(nn.Module):
    def __init__(self, in_channel_list, target_channel_list, conv_type):
        super(SingleResolutionNeck, self).__init__()
        self.lateral_convs = nn.ModuleList()
        assert len(in_channel_list) % 2 == 1, "to use single resolution neck, the number of input channels must be odd"
        for in_channels, target_channels in (in_channel_list):
            self.lateral_convs.append(conv_type(in_channels, target_channels, kernel_size=1))
    def forward(self, ):
        pass
        
    

class LightWeightDetector(nn.Module):
    def __init__(self, backbone, hook_list, nclasses, finetune=False):
        super(LightWeightDetector, self).__init__()
        self.nclasses = nclasses
        self.backbone = backbone
        self.hook_list = hook_list
        self.finetune = finetune
        if not self.finetune:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone.train()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        self.backbone.activations.clear()
        _ = self.backbone(x)
        activations = self.backbone.activations
        print([i.shape for i in activations])


if __name__ == "__main__":
    import torchvision
    backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
    print(backbone)


