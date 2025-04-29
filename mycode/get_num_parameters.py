import torch
import torchvision.models as models

model_names = [
    'alexnet',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s'
]

for name in model_names:
    model = getattr(models, name)(pretrained=False)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:10s} • total params: {num_params:,} • trainable params: {num_trainable:,}")
