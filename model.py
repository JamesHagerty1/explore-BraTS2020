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
		print(x.shape)
		(x, concat_x) = self.encoder(x)
		x = self.decoder(x, concat_x)
		print(x.shape)
		return x
	

class Encoder(nn.Module):
	def __init__(self, conv_channels, conv_kernel_size, pool_kernel_size):
		super().__init__()
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(conv_channels[i], conv_kernel_size) 
                for i in range(len(conv_channels))])
		self.pool = nn.MaxPool3d(pool_kernel_size, stride=pool_kernel_size)

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
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose3d(upconv_channels[i], upconv_channels[i], 
		       	upconv_kernel_size, stride=upconv_kernel_size)
    			for i in range(len(upconv_channels))])
		self.conv_blocks = nn.ModuleList(
			[ConvBlock(conv_channels[i], conv_kernel_size) 
                for i in range(len(conv_channels))])

	def U_concat(self, x, x_u):
		# Per channel, crop center cuboid from x_u with dims matching x cuboid
		(_, _, x_d1, x_d2, x_d3) = x.shape
		(_, _, x_u_d1, x_u_d2, x_u_d3) = x_u.shape
		# TBD, check if x cuboid always smaller than x_u cuboid
		if (x_d1 > x_u_d1 or x_d2 > x_u_d2 or x_d3 > x_u_d3):
			raise ValueError("Invalid U_concat")
		(o1, o2, o3) = \
			((x_u_d1 - x_d1) // 2, (x_u_d1 - x_d1) // 2, (x_u_d1 - x_d1) // 2) 
		x_u = x_u[:, :, o1:o1+x_d1, o2:o2+x_d2, o3:o3+x_d3]
		x = torch.cat([x, x_u], dim=1)
		return x
		
	def forward(self, x, concat_x):
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
	