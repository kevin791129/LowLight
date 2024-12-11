import torch
import torch.nn as nn
from loss import LossFunction


def rgb2yCbCr(input_im):
    im_flat = input_im.contiguous().permute((0, 2, 3, 1)).reshape(-1, 3)
    mat = torch.Tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).cuda() / 255.0
    bias = torch.Tensor([16.0, 128.0, 128.0]).cuda() / 255.0
    temp = im_flat.mm(mat) + bias
    out = temp.reshape(input_im.shape[0], input_im.shape[2], input_im.shape[3], 3).permute((0, 3, 1, 2))
    return out

def yCbCr2rbg(input_im):
    im_flat = input_im.contiguous().permute((0, 2, 3, 1)).reshape(-1, 3)
    mat = torch.Tensor([[255.0 / 219.0, 255.0 / 219.0, 255.0 / 219.0], [0.0, -255.0 / 224.0 * 1.772 * 0.114 / 0.587, 255.0 / 224.0 * 1.772], [255.0 / 224.0 * 1.402, -255.0 / 224.0 * 1.402 * 0.299 / 0.587, 0.0]]).cuda()
    bias = torch.Tensor([16.0, 128.0, 128.0]).cuda() / 255.0
    temp = (im_flat - bias).mm(mat)
    out = temp.reshape(input_im.shape[0], input_im.shape[2], input_im.shape[3], 3).permute((0, 3, 1, 2))
    return out

def rgb2hsl(input_im):
    cmax, cmax_idx = torch.max(input_im, dim=1, keepdim=True)
    cmin = torch.min(input_im, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(input_im[:, 0:1, :, :]).cuda()
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((input_im[:, 1:2] - input_im[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((input_im[:, 2:3] - input_im[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((input_im[:, 0:1] - input_im[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.
    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h).cuda()
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)

def hsl2rgb(input_im):
    hsl_h, hsl_s, hsl_l = input_im[:, 0:1], input_im[:, 1:2], input_im[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(input_im).cuda()
    _o = torch.zeros_like(_c).cuda()
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

    
class IlluminationNetwork(nn.Module):
    def __init__(self, layers, channels, in_channels=1):
        super(IlluminationNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input[:, [0], :, :]
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class SelfCalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(SelfCalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class TrainModel(nn.Module):

    def __init__(self, stage=3, color="hsl", in_channels=1):
        super(TrainModel, self).__init__()
        self.stage = stage
        self.color = "ycbcr" if color.lower() == "ycbcr" else "hsl"
        self.in_channels = in_channels
        self.illumination = IlluminationNetwork(layers=1, channels=3, in_channels=in_channels)
        self.self_calibrate = SelfCalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()
        self.rgb2other = rgb2hsl if self.color == "hsl" else rgb2yCbCr
        self.other2rbg = hsl2rgb if self.color == "hsl" else yCbCr2rbg
        self.channel_idx = 2 if self.color == "hsl" else 0

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        color = self.rgb2other(input)
        input_channel = color.select(1, self.channel_idx).unsqueeze(1)
        input_op = input_channel
        for i in range(self.stage):
            if self.in_channels > 1:
                if self.color == "hsl":
                    input_other_channel = rgb2yCbCr(input if len(rlist) == 0 else rlist[-1]).select(1, 0).unsqueeze(1)
                else:
                    input_other_channel = rgb2hsl(input if len(rlist) == 0 else rlist[-1]).select(1, 2).unsqueeze(1)
                illu_input = torch.cat((input_op, input_other_channel), 1)
            else:
                illu_input = input_op
            inlist.append(input_op)
            i = self.illumination(illu_input)
            r = input_channel / i
            r = torch.clamp(r, 0, 1)
            att = self.self_calibrate(r)
            input_op = input_channel + att
            ilist.append(i)
            temp = color.clone()
            temp[:, [self.channel_idx], :, :] = r
            temp = self.other2rbg(temp)
            rlist.append(temp)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss


class PredictModel(nn.Module):

    def __init__(self, weights, color="hsl", in_channels=1):
        super(PredictModel, self).__init__()
        self.color = "ycbcr" if color.lower() == "ycbcr" else "hsl"
        self.in_channels = in_channels
        self.illumination = IlluminationNetwork(layers=1, channels=3, in_channels=in_channels)
        self._criterion = LossFunction()
        self.rgb2other = rgb2hsl if self.color == "hsl" else rgb2yCbCr
        self.other2rbg = hsl2rgb if self.color == "hsl" else yCbCr2rbg
        self.channel_idx = 2 if self.color == "hsl" else 0

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        color = self.rgb2other(input)
        input_channel = color.select(1, self.channel_idx).unsqueeze(1)
        if self.in_channels > 1:
            if self.color == "hsl":
                input_other_channel = rgb2yCbCr(input).select(1, 0).unsqueeze(1) 
            else:
                input_other_channel = rgb2hsl(input).select(1, 2).unsqueeze(1) 
            illu_input = torch.cat((input_channel, input_other_channel), 1)
        else:
            illu_input = input_channel
        i = self.illumination(illu_input)
        r = input_channel / i
        r = torch.clamp(r, 0, 1)
        color[:, [self.channel_idx], :, :] = r
        color = self.other2rbg(color)
        return i, color

    def _loss(self, input):
        i, _ = self(input)
        loss = self._criterion(input, i)
        return loss
