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
		skip_layer_inputs = []
		for conv_block in self.conv_blocks:
			x = conv_block(x)
			skip_layer_inputs.append(x)
			break
		return skip_layer_inputs


class ConvBlock(nn.Module):
	def __init__(self, channels, conv_kernel_size):
		super().__init__()
		self.channels = channels
		self.convs = nn.ModuleList(
			[nn.Conv3d(channels[i], channels[i+1], conv_kernel_size)
                for i in range(len(channels)-1)])
        # ?? Batch norm
		self.relus = nn.ModuleList([nn.ReLU() for _ in range(len(channels)-1)])
		
	def forward(self, x):
		print("ConvBlock()")
		for i in range(len(self.channels)-1):
			print(self.convs[i])
			print(x.shape)
			x = self.convs[i](x)
			x = self.relus[i](x)
			print(x.shape)
		return x
	