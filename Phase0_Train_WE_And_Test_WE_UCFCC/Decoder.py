import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_FPN.BasicConv2d import BasicConv2d

class Decoder(nn.Module):
    def __init__(self, IF_BN=True, leaky_relu=False, is_aspp=False, n_stack=1):
        super(Decoder, self).__init__()
        self.is_aspp = is_aspp
        self.n_stack = n_stack
        self.Decoder_Block_1 = nn.Sequential(
            BasicConv2d(2048, 512, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN))

        self.Decoder_Block_2 = nn.Sequential(
            BasicConv2d(512, 256, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(256, 256, 3, 1, 1, if_Bn=IF_BN))

        self.Decoder_Block_3 = nn.Sequential(
            BasicConv2d(256, 64, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(64, 64, 3, 1, 1, if_Bn=IF_BN),
            BasicConv2d(64, 32, 3, 1, 1, if_Bn=IF_BN),
        )

        if self.is_aspp:
            for stack_i in range(n_stack):
                setattr(self, 'aspp_layer_{:d}'.format(stack_i), nn.ModuleList(aspp(in_channel=32)))

        # additional layers specific for Phase 3
        # style one: 32 --> 128 --> GAP
        self.pred_conv = nn.Conv2d(32, 128, 3, padding=1)
        self.pred_bn = nn.BatchNorm2d(128)
        self.GAP = nn.AdaptiveAvgPool2d(1)


        if not leaky_relu:
            self.output = nn.Sequential(
                nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(inplace=True))
        else:
            self.output = nn.Sequential(
                nn.Conv2d(32, 1, 1, 1, 0), nn.LeakyReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # when attn_weights=None, means that it is in testing phase
    def forward(self, B5_C3, B4_C3, B3_C3, B2_C2):
        link_1 = functional.interpolate(
            self.Decoder_Block_1(B5_C3),
            size=B4_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B4_C3
        link_2 = functional.interpolate(
            self.Decoder_Block_2(link_1),
            size=B3_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B3_C3
        link_3 = functional.interpolate(
            self.Decoder_Block_3(link_2),
            size=B2_C2.shape[2:4],
            mode="bilinear",
            align_corners=True) + B2_C2

        x = link_3
        pred_attn = self.GAP(functional.relu_(self.pred_bn(self.pred_conv(link_3))))
        pred_attn = functional.softmax(pred_attn ,dim=1)
        pred_attn_list = torch.chunk(pred_attn, 4, dim=1)

        if self.is_aspp:
            aspp_out = []
            for stack_i in range(self.n_stack):
                cur_aspp = getattr(self, 'aspp_layer_{:d}'.format(stack_i))
                for k, v in enumerate(cur_aspp):
                    if k%2 == 0:
                        aspp_out.append(functional.relu_(cur_aspp[k+1](v(x)) * 0.25)) # for phase 3, I move '0.25' and 'relu_' here
                    else:
                        continue

            # add weights in each channel of every branch
            # I suspect, we should only train the layers added in Phase 3, after that,
            # Then we train together for refining
            for i in range(4):
                x = x + aspp_out[i] * pred_attn_list[i]


        return self.output(x), pred_attn

def aspp(aspp_num=4, aspp_stride=2, in_channel=512, use_bn=True):
    aspp_list = []
    for i in range(aspp_num):
        pad = (i+1) * aspp_stride
        dilate = pad
        conv_aspp = nn.Conv2d(in_channel, in_channel, 3, padding=pad, dilation=dilate)
        aspp_list.append(conv_aspp)
        if use_bn:
            aspp_list.append(nn.BatchNorm2d(in_channel))

    return aspp_list
