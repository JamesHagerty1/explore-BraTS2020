import torch
import torch.nn as nn


################################################################################


class UNet3D(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.encoder = Encoder(c.enc_channels, c.conv_kernel_size,
			c.pool_kernel_size)
		
	def forward(self, x):
		enc_features = self.encoder(x)
		return x
	

class Encoder(nn.Module):
	def __init__(self, channels, conv_kernel_size, pool_kernel_size):
		super().__init__()
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(channels[i], conv_kernel_size) 
                for i in range(len(channels))])
		self.pool = nn.MaxPool3d(pool_kernel_size)

	def forward(self, x):
		return x


class ConvBlock(nn.Module):
	def __init__(self, channels, conv_kernel_size):
		super().__init__()
		print("hii")
		# self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
		# self.relu = nn.ReLU()
		# self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size)
		
	def forward(self, x):
		# x = self.conv1(x)
		# x = self.relu(x)
		# x = self.conv2(x)
		return x
	