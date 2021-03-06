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


        if not leaky_relu:
            self.output = nn.Sequential(
                nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(inplace=True))
        else:
            self.output = nn.Sequential(
                nn.Conv2d(32, 1, 1, 1, 0)) # just remove relu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # for this special repo, add an additional zero_index in
    # zero_index range: -1, 0, ..., num_branch*C
    def forward(self, B5_C3, B4_C3, B3_C3, B2_C2, zero_index, weight_scale):
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
        if self.is_aspp:
            original_weight_tmp = None

            aspp_out = []
            for stack_i in range(self.n_stack):
                cur_aspp = getattr(self, 'aspp_layer_{:d}'.format(stack_i))
                for k, v in enumerate(cur_aspp):
                    if k%2 == 0:
                        aspp_out.append(cur_aspp[k+1](v(x)))
                    else:
                        continue


            # Original performance
            if zero_index == -1:
                pass
            # zero out the index(channel-wise, branch-wise) to be 0
            else:
                cur_b = zero_index // 32
                cur_c = zero_index % 32
                mask = torch.ones_like(x)
                mask[:, cur_c, :, :] = mask[:, cur_c, :, :] * weight_scale # weight_scale should be 0 or 1.1
                # only zero out the feature
                aspp_out[cur_b] = aspp_out[cur_b] * mask

            for i in range(4):
                x = x + aspp_out[i] * 0.25

            x = functional.relu_(x)


        return self.output(x)

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
