import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InceptionConvModel(nn.Module):
 ## the primary difference lies in how the inception units are organized and parameter sharing, rather than a difference in the individual inception unit architectures. In "Use  same unit," there's parameter sharing across passes, while in "Use with different unit," each pass has its own set of parameters despite the units having the same internal architecture.  
  # inception_passes defines how many times the inception module is being passed through
  def __init__(self, inception_passes, n_channels_input, use_same_unit=True, **kwargs):
    super().__init__()
    
    self.inception_passes = inception_passes
    self.n_channels_input = n_channels_input
    assert self.n_channels_input % 4 == 0, f"Provide n_channels_input that is divisable by 4. n_channels_input received: {self.n_channels_input}"
    self.n_channels_branch = self.n_channels_input // 4
    self.batchnorm1 = nn.BatchNorm3d(32)
    self.batchnorm2 = nn.BatchNorm3d(64)
    self.activation = nn.ReLU()
    self._1x1x1_conv_1 = None
    self._1x1X1_conv_2 = None
    self._3x3x3_conv_2 = None
    self._1x1x1_conv_3 = None
    self._5x5x5_conv_3 = None
    self._3x3x3_max_pooling = None
    self._1x1x1_conv_4 = None
    self.define_filters(n_channels_input)
  
  def define_filters(self, num_channels):
    # stride = 1
    kernel_size = 1
    same = (kernel_size - 1) // 2 
    self._1x1x1_conv_1 = nn.Conv3d(num_channels, self.n_channels_branch, kernel_size=(1, 1, 1), padding=same).to(device)
    
   # stride = 1
    kernel_size = 1
    same =(kernel_size - 1) // 2 
    self._1x1x1_conv_2 = nn.Conv3d(num_channels, 64, kernel_size=(1, 1, 1), padding=same).to(device)
   # stride = 1
    kernel_size = 3 
    same = (kernel_size - 1) // 2 
    self._3x3x3_conv_2 = nn.Conv3d(64, self.n_channels_branch, kernel_size=(3, 3, 3), padding=same).to(device)

   # stride = 1
    kernel_size = 1
    same = (kernel_size - 1) // 2 
    self._1x1x1_conv_3 = nn.Conv3d(num_channels, 64, kernel_size=(1, 1, 1), padding=same).to(device)
   # stride = 1
    kernel_size = 5 
    same = (kernel_size - 1) // 2 
    self._5x5x5_conv_3 = nn.Conv3d(64, self.n_channels_branch, kernel_size=(5, 5, 5), padding=same).to(device)
    
   # stride = 1
    kernel_size = 3 
    same = (kernel_size - 1) // 2 
    self._3x3x3_max_pooling = nn.MaxPool3d(kernel_size=3, stride=1, padding=same).to(device)
   # stride = 1
    kernel_size = 1 
    same = (kernel_size - 1) // 2 
    self._1x1x1_conv_4 = nn.Conv3d(num_channels, self.n_channels_branch, kernel_size=(1, 1, 1), padding=same).to(device)

  def forward(self, inputs):
    x = inputs.to(device)
    for i in range(self.inception_passes):
      print(f'Iteration {i} start')
      if i > 0:
        self.define_filters(x.shape[1])
      b1 = self._1x1x1_conv_1(x)
      b1 = self.batchnorm1(b1)
      b1 = self.activation(b1)

      b2 = self._1x1x1_conv_2(x)
      b2 = self.batchnorm2(b2)
      b2 = self.activation(b2)
      b2 = self._3x3x3_conv_2(b2)
      b2 = self.batchnorm1(b2)
      b2 = self.activation(b2)
      
      b3 = self._1x1x1_conv_3(x)
      b3 = self.batchnorm2(b3)
      b3 = self.activation(b3)
      b3 = self._5x5x5_conv_3(b3)
      b3 = self.batchnorm1(b3)
      b3 = self.activation(b3)

      b4 = self._3x3x3_max_pooling(x)
      b4 = self._1x1x1_conv_4(b4)
      b4 = self.batchnorm1(b4)
      b4 = self.activation(b4)

      x = torch.cat((b1, b2, b3, b4), 1).to(device)
      print(x.shape)
    return x

# Test Input
#inputs = torch.randn(8, 128, 149, 10, 10).to(device)
#model = InceptionConvModel(3, 128).to(device)
#output = model(inputs).to(device)
