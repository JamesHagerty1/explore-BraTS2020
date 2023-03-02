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
		self.final_conv = nn.Conv3d(c.dec_conv_channels[-1][-1], c.num_labels, 1)

	def forward(self, x):
		# print("UNet3D()")
		# print(x.shape)
		(x, concat_x) = self.encoder(x)
		x = self.decoder(x, concat_x)
		# x = self.final_conv(x)				# UNSURE OF THIS
		# print(x.shape)
		return x
	

class Encoder(nn.Module):
	def __init__(self, conv_channels, conv_kernel_size, pool_kernel_size):
		super().__init__()
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(conv_channels[i], conv_kernel_size) 
                for i in range(len(conv_channels))])
		self.pool = nn.MaxPool3d(pool_kernel_size, stride=pool_kernel_size)

	def forward(self, x):
		# print("Encoder()")
		save = [] 
		for conv_block in self.conv_blocks:
			# print("ConvBlock + MaxPool3D")
			# print(x.shape)
			x = conv_block(x)
			# print(x.shape)
			save.append(x)
			x = self.pool(x)
			# print(x.shape)
		return (save[-1], save[-2::-1])


class Decoder(nn.Module):
	def __init__(self, upconv_channels, upconv_kernel_size, conv_channels, 
	    conv_kernel_size):
		super().__init__()
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose3d(upconv_channels[i], upconv_channels[i], 
		       	upconv_kernel_size, stride=upconv_kernel_size)
    			for i in range(len(upconv_channels))])
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(conv_channels[i], conv_kernel_size) 
                for i in range(len(conv_channels))])

	# TBD, assert x size < x_u size always
	def U_concat(self, x, x_u):
		(_, _, d1, d2, d3) = x_u.shape
		x = nn.functional.interpolate(x, (d1, d2, d3))
		x = torch.cat([x, x_u], dim=1)
		return x
		
	def forward(self, x, concat_x):
		# print("Decoder()")
		for i, x_u in enumerate(concat_x):
			x = self.upconvs[i](x)
			x = self.U_concat(x, x_u)
			x = self.conv_blocks[i](x)
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
	