import torch.nn as nn
import lightningaddon as la


def conv_bn_relu(in_shape, ks, stride=1, padding=1, diff_out=None):
    diff_out = in_shape if diff_out == None else diff_out
    return nn.Sequential(
        nn.Conv2d(in_shape, diff_out, (ks, ks), (stride, stride), (padding, padding)),
        nn.BatchNorm2d(diff_out),
        nn.ReLU(inplace=True),
    )


def pooler(ks=2, stride=2):
    return nn.MaxPool2d((ks, ks), (stride, stride))


def fc(num_classes):
    return nn.Sequential(
        la.avgpoolflatten(),
        nn.Linear(512, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
    )


def conv_layers(in_shape, out_shape, ks=3, stride=1, num_repeats=1):
    output_layers = [conv_bn_relu(in_shape, ks, stride) for _ in range(num_repeats)]
    if in_shape != out_shape:
        output_layers.append(conv_bn_relu(in_shape, ks, stride, diff_out=out_shape))
    else:
        output_layers.append(conv_bn_relu(in_shape, ks, stride))
    return nn.Sequential(*output_layers)


def expand_layers(layer_list, num_classes):
    final_ex = []
    for layer in layer_list:
        if layer[0] == "M":
            final_ex.append(pooler())
        elif layer[0] == "F":
            final_ex.append(fc(num_classes))
        elif layer[0] == "S":
            final_ex.append(nn.Softmax2d())
        else:
            ins, outs, ks, stride, num_repeats = (
                layer[1],
                layer[2],
                layer[0],
                1,
                layer[3],
            )

            final_ex.append(conv_layers(ins, outs, ks, stride, num_repeats))
    return nn.Sequential(*final_ex)


def vgg_base(num_classes, nc=3, name="vgg_16"):
    d_expansion = {
        "vgg_13": 2,
        "vgg_16": 3,
        "vgg_19": 4,
    }
    expanded = d_expansion[name]

    return expand_layers(
        [
            # ks, in, out, num_repeats
            [3, nc, 64, 1],
            [3, 64, 64, 1],
            "M",
            [3, 64, 128, 2],
            "M",
            [3, 128, 256, 2],
            "M",
            [3, 256, 512, 2],
            "M",
            [3, 512, 512, expanded],
            "M",
            [3, 512, 512, expanded],
            "F",
        ],
        num_classes=num_classes,
    )


def vgg_13(num_classes, nc=3):
    return vgg_base(num_classes, nc, "vgg_13")


def vgg_16(num_classes, nc=3):
    return vgg_base(num_classes, nc, "vgg_16")


def vgg_19(num_classes, nc=3):
    return vgg_base(num_classes, nc, "vgg_19")


#  import torch
#  mo1 = vgg_13(3,3)
#  mo2 = vgg_16(3,3)
#  mo3 = vgg_19(3,3)
#
#  ra = torch.randn(1, 3, 64, 64)
#  mo1(ra)
#  mo2(ra)
#  mo3(ra)
