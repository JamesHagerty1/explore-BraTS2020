import torch
import torch.nn as nn


################################################################################


class UNet3D(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.encoder = Encoder(c.enc_channels, c.conv_kernel_size,
			c.pool_kernel_size)
		
	def forward(self, x):
		(x, skip_layer_inputs) = self.encoder(x)
		print(x.shape)
		print([res.shape for res in skip_layer_inputs])
		return x
	

class Encoder(nn.Module):
	def __init__(self, channels, conv_kernel_size, pool_kernel_size):
		super().__init__()
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(channels[i], conv_kernel_size) 
                for i in range(len(channels))])
		self.pool = nn.MaxPool3d(pool_kernel_size)

	def forward(self, x):
		skip_layer_inputs = []
		for conv_block in self.conv_blocks:
			x = conv_block(x)
			skip_layer_inputs.append(x)
			x = self.pool(x)
		return skip_layer_inputs[-1], skip_layer_inputs[-2::-1]


class ConvBlock(nn.Module):
	def __init__(self, channels, conv_kernel_size):
		super().__init__()
		self.channels = channels
		self.convs = nn.ModuleList(
			[nn.Conv3d(channels[i], channels[i+1], conv_kernel_size)
                for i in range(len(channels)-1)])
		self.norms = nn.ModuleList(
			[nn.BatchNorm3d(channels[i]) 
    			for i in range(1, len(channels))])
		self.relus = nn.ModuleList(
			[nn.ReLU() 
    			for _ in range(len(channels)-1)])
		
	def forward(self, x):
		for i in range(len(self.channels)-1):
			x = self.convs[i](x)
			x = self.norms[i](x)
			x = self.relus[i](x)
		return x
	