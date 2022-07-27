import torch.nn as nn
import torch.nn.functional as F
import lightningaddon as la


def conv_bn_relu(in_shape, ks=3, stride=1, dirate=1, diff_out=None):
    diff_out = in_shape if diff_out == None else diff_out
    return nn.Sequential(
        nn.Conv2d(
            in_shape,
            diff_out,
            (ks, ks),
            (stride, stride),
            (dirate, dirate),
            (dirate, dirate),
        ),
        nn.BatchNorm2d(diff_out),
        nn.ReLU(inplace=True),
    )


def _upsample_like(inp, out):
    inp = F.upsample(inp, size=out.shape[2:], mode="bilinear")
    return inp


def pooler(ins, outs, dirate=1):
    return nn.Sequential(
        conv_bn_relu(ins, dirate=dirate, diff_out=outs),
        nn.MaxPool2d(2, stride=2, ceil_mode=True),
    )


def expand_layers(layer_list):
    return_list = []
    for i in layer_list:
        inch, outch, dirate, n_repeats = i
        if i[0] == "p":
            return_list.append(pooler(inch, outch, dirate))
        else:
            return_list.extend(
                [
                    conv_bn_relu(inch, diff_out=outch, dirate=dirate)
                    for _ in range(n_repeats)
                ]
            )
    return nn.Sequential(*return_list)


def RSU7(ins=3, mids=12, outs=3):
    out_list = [conv_bn_relu(ins, 3, 1, 1, outs)]
    out_list.append(
        expand_layers(
            [
                # ins, outs, dirate, repeat
                ("p", outs, 1, 1)
            ]
        )
    )


#  import torch
#  mo1 = vgg_13(3,3)
#  mo2 = vgg_16(3,3)
#  mo3 = vgg_19(3,3)
#
#  ra = torch.randn(1, 3, 64, 64)
#  mo1(ra)
#  mo2(ra)
#  mo3(ra)
