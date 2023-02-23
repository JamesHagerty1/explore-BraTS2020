import torch
import torch.nn as nn


################################################################################


class UNet3D(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.encoder = Encoder(c.enc_conv_channels, c.conv_kernel_size,
			c.pool_kernel_size)
		self.decoder = Decoder(c.dec_upconv_channels, c.upconv_kernel_size, 
			c.dec_conv_channels, c.conv_kernel_size)

	def forward(self, x):
		(x, concat_x) = self.encoder(x)
		x = self.decoder(x, concat_x)
		return x
	

class Encoder(nn.Module):
	def __init__(self, conv_channels, conv_kernel_size, pool_kernel_size):
		super().__init__()
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(conv_channels[i], conv_kernel_size) 
                for i in range(len(conv_channels))])
		self.pool = nn.MaxPool3d(pool_kernel_size)

	def forward(self, x):
		save = [] 
		for conv_block in self.conv_blocks:
			x = conv_block(x)
			save.append(x)
			x = self.pool(x)
		return (save[-1], save[-2::-1])


class Decoder(nn.Module):
	def __init__(self, upconv_channels, upconv_kernel_size, conv_channels, 
	    conv_kernel_size):
		super().__init__()
		print(upconv_channels, upconv_kernel_size, conv_channels, conv_kernel_size)
		

	def U_concat(self):
		pass

	def forward(self, x, concat_x):
		return x


class ConvBlock(nn.Module):
	def __init__(self, conv_channels, conv_kernel_size):
		super().__init__()
		self.conv_channels = conv_channels
		self.convs = nn.ModuleList(
			[nn.Conv3d(conv_channels[i], conv_channels[i+1], conv_kernel_size)
                for i in range(len(conv_channels)-1)])
		self.norms = nn.ModuleList(
			[nn.BatchNorm3d(conv_channels[i]) 
    			for i in range(1, len(conv_channels))])
		self.relus = nn.ModuleList(
			[nn.ReLU() 
    			for _ in range(len(conv_channels)-1)])
		
	def forward(self, x):
		for i in range(len(self.conv_channels)-1):
			x = self.convs[i](x)
			x = self.norms[i](x)
			x = self.relus[i](x)
		return x
	