import torch
from torch import nn
import torchvision

# This module uses Conv3D blocks in order to extract meaningful features from the input image, following 
# the output into an InceptionConvModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackboneConvModel(nn.Module):

  def __init__(self, start_kernel_size = 7, n_layers = 3, decrease_kernel = True, final_channels = 128,
               intermediary_channels = 64, **kwargs):
    super().__init__(**kwargs)
    kernel_size = start_kernel_size
    self.kernel_size_list = []
    if decrease_kernel:
      decrease_amount = max(1, start_kernel_size // n_layers)
    else:
      decrease_amount = 0
    for _ in range(n_layers):
      self.kernel_size_list.append(kernel_size)
      kernel_size = kernel_size - decrease_amount
      if kernel_size < 1:
        kernel_size = 1
    
    layers = []
    for idx, kernel_size in enumerate(self.kernel_size_list):
      layers.append(nn.Conv3d(3 if idx == 0 else intermediary_channels,
               intermediary_channels if idx != len(self.kernel_size_list)-1 else final_channels,
               kernel_size,
               stride = 2 if idx < len(self.kernel_size_list) // 2 else 1))
      layers.append(nn.ReLU())
    
    self.start_layers = nn.Sequential(*layers)

  def forward(self, inputs):
    return self.start_layers(inputs) 

# Test input
#backbone = BackboneConvModel().to(device)
#inputs = torch.rand(8, 3, 149, 224, 224).to(device)
#outputs = print(backbone(inputs).shape)

