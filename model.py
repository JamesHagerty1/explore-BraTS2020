import torch
import torch.nn as nn


################################################################################


class UNet3D(nn.Module):
	def __init__(self, c):
		super().__init__()
		
	def forward(self, x):
		print(x.shape)
		return x
	