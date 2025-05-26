import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from models.blocks.conv_blocks import SeparableConv, DoubleConv, ResidualBlock, DenseBlock, InceptionBlock


class SingleHeadAttention