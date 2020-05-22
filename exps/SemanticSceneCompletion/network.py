# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from config import config
from resnet import get_resnet50

class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

class CVAE(nn.Module):
    def __init__(self, norm_layer, bn_momentum, latent_size=16):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.mean = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)
        self.log_var = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)

        self.decoder_x = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.latent_size*2, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(self.latent_size, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Dropout3d(0.1),
            nn.Conv3d(self.latent_size, 2, kernel_size=1, bias=True)
        )

    def forward(self, x, gt=None):
        b, c, h, w, l = x.shape
        if self.training:
            gt = gt.view(b, 1, h, w, l).float()
            for_encoder = torch.cat([x, gt], dim=1)
            enc = self.encoder(for_encoder)
            pred_mean = self.mean(enc)
            pred_log_var = self.log_var(enc)
            decoder_x = self.decoder_x(x)
            out_samples = []
            out_samples_gsnn = []
            for i in range(config.samples):
                std = pred_log_var.mul(0.5).exp_()
                eps = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                z1 = eps * std + pred_mean
                z2 = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                edge = self.decoder(torch.cat([decoder_x, z1], dim=1))
                out_samples.append(edge)
                edge_gsnn = self.decoder(torch.cat([decoder_x, z2], dim=1))
                out_samples_gsnn.append(edge_gsnn)
            edge = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            edge = torch.mean(edge, dim=0)
            edge_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples_gsnn])
            edge_gsnn = torch.mean(edge_gsnn, dim=0)
            return pred_mean, pred_log_var, edge_gsnn, edge
        else:
            out_samples = []
            for i in range(config.samples):
                z = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                decoder_x = self.decoder_x(x)
                out = self.decoder(torch.cat([decoder_x, z], dim=1))
                out_samples.append(out)
            edge_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            edge_gsnn = torch.mean(edge_gsnn, dim=0)
            return None, None, edge_gsnn, None

class STAGE1(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE1, self).__init__()
        self.business_layer = []
        self.oper1 = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper1)
        self.completion_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer1)
        self.completion_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer2)
        self.cvae = CVAE(norm_layer=norm_layer, bn_momentum=bn_momentum, latent_size=config.lantent_size)
        self.business_layer.append(self.cvae)
        self.classify_edge = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, 2, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_edge)

    def forward(self, ttt, mmm, eee=None):
        ttt = self.oper1(ttt)
        c1 = self.completion_layer1(ttt)
        c2 = self.completion_layer2(c1)
        up_edge1 = self.classify_edge[0](c2)
        up_edge1 = up_edge1 + c1
        up_edge2 = self.classify_edge[1](up_edge1)
        pred_edge_raw = self.classify_edge[2](up_edge2)
        _, pred_edge_binary = torch.max(pred_edge_raw, dim=1, keepdim=True)
        pred_mean, pred_log_var, pred_edge_gsnn, pred_edge= self.cvae(pred_edge_binary.float(), eee)
        return pred_edge_raw, pred_edge_gsnn, pred_edge, pred_mean, pred_log_var


class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []
        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)
        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)
        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)
        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_semantic)
        self.oper_edge = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.oper_edge_cvae = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper_edge)
        self.business_layer.append(self.oper_edge_cvae)

    def forward(self, f2, mmm, raw, gsnn):
        if self.resnet_out != self.feature:
            f2 = self.downsample(f2)
        f2 = F.interpolate(f2, scale_factor=16, mode='bilinear', align_corners=True)
        b, c, h, w = f2.shape
        f2 = f2.view(b, c, h * w).permute(0, 2, 1)
        vecs = torch.zeros(b, 1, c).cuda()
        segVec = torch.cat((f2, vecs), 1)
        segres = [torch.index_select(segVec[i], 0, mmm[i]) for i in range(b)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)
        edge_proi = self.oper_edge(raw)
        edge_proi_gsnn = self.oper_edge_cvae(gsnn)
        seg_fea = segres + edge_proi + edge_proi_gsnn
        semantic1 = self.semantic_layer1(seg_fea)
        semantic2 = self.semantic_layer2(semantic1)
        up_sem1 = self.classify_semantic[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic[1](up_sem1)
        pred_semantic = self.classify_semantic[2](up_sem2)

        return pred_semantic, None

class Network(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network, self).__init__()
        self.business_layer = []

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=norm_layer)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.stage1 = STAGE1(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage1.business_layer
        self.stage2 = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer

    def forward(self, rrr, mmm, ttt, eeee=None, eee=None):
        f2 = self.backbone(rrr)
        raw, gsnn, pred_edge, pred_mean, pred_log_var = self.stage1(ttt, mmm, eee)
        pp, ppp = self.stage2(f2, mmm, raw, gsnn)

        if self.training:
            return pp, ppp, raw, gsnn, pred_edge, pred_mean, pred_log_var
        return pp, ppp, gsnn

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    pass