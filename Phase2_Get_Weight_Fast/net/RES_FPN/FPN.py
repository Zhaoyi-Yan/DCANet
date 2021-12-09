import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_FPN.Encoder import Encoder
from net.RES_FPN.BasicConv2d import BasicConv2d
from net.RES_FPN.Decoder import Decoder


class FPN(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, leaky_relu=False, is_aspp=False, n_stack=1):
        super(FPN, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = Decoder(IF_BN=True, leaky_relu=leaky_relu, is_aspp=is_aspp, n_stack=n_stack)

    def forward(self, x, zero_index=-1, weight_scale=0.0):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2, zero_index, weight_scale=weight_scale)
        return output
