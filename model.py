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

class IlluminationNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(IlluminationNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
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

        illu = fea + input
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

    def __init__(self, stage=3):
        super(TrainModel, self).__init__()
        self.stage = stage
        self.illumination = IlluminationNetwork(layers=1, channels=3)
        self.self_calibrate = SelfCalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()
        self.rgb2yCbCr = rgb2yCbCr
        self.yCbCr2rbg = yCbCr2rbg

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        yCbCr = self.rgb2yCbCr(input)
        input_y = yCbCr.select(1, 0).unsqueeze(1)
        input_op = input_y
        for i in range(self.stage):
            # temp = yCbCr.clone()
            # temp[:, 0, :, :] = input_op
            # inlist.append(yCbCr2rbg(temp))
            inlist.append(input_op)
            i = self.illumination(input_op)
            r = input_y / i
            r = torch.clamp(r, 0, 1)
            att = self.self_calibrate(r)
            input_op = input_y + att
            # temp = yCbCr.clone()
            # temp[:, 0, :, :] = i
            # ilist.append(yCbCr2rbg(temp))
            ilist.append(i)
            temp = yCbCr.clone()
            temp[:, 0, :, :] = r
            rlist.append(yCbCr2rbg(temp))
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss



class PredictModel(nn.Module):

    def __init__(self, weights):
        super(PredictModel, self).__init__()
        self.illumination = IlluminationNetwork(layers=1, channels=3)
        self._criterion = LossFunction()
        self.rgb2yCbCr = rgb2yCbCr
        self.yCbCr2rbg = yCbCr2rbg

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
        yCbCr = self.rgb2yCbCr(input)
        input_y = yCbCr.select(1, 0).unsqueeze(1)
        i = self.illumination(input_y)
        r = input_y / i
        r = torch.clamp(r, 0, 1)
        temp = yCbCr
        temp[:, 0, :, :] = r
        return i, yCbCr2rbg(temp)


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

