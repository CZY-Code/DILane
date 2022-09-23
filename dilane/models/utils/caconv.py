import torch
import torch.nn.functional as F
import copy
import torch.nn as nn
from dilane.models.utils.dynamic_conv import Dynamic_conv2d
from mmcv.cnn import ConvModule


class caConv(nn.Module):
    def __init__(self, fc_hidden_dim, num_priors, sample_points, refine_layers):
        super().__init__()
        self.fc_hidden_dim = fc_hidden_dim
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.stage = refine_layers
        # self.dynamic_conv =  nn.Sequential(
        #     Dynamic_conv2d(in_planes=64, out_planes=64, kernel_size_h=9, kernel_size_w=2, padding=(4, 0)),
        #     nn.BatchNorm2d(self.fc_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     Dynamic_conv2d(in_planes=64, out_planes=64, kernel_size_h=9, kernel_size_w=2, padding=(4, 0)),
        #     nn.BatchNorm2d(self.fc_hidden_dim),
        #     nn.ReLU(inplace=True),
        # )
        self.cala_weight_fc = nn.Sequential(
            nn.Linear(self.fc_hidden_dim*self.stage, (self.fc_hidden_dim*self.stage)//4),
            nn.ReLU(inplace=True),
            nn.Linear((self.fc_hidden_dim*self.stage)//4, self.stage),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=-1)
        )

        self.cala_weight_conv = nn.ModuleList()
        for i in range(self.stage):
            self.cala_weight_conv.append(
                ConvModule(self.fc_hidden_dim,
                    self.fc_hidden_dim, 
                    (3, 3),
                    padding=(1, 1),
                    bias=False,
                    norm_cfg=dict(type='BN')),
            )
        self.Caconv_1 = nn.Sequential(
            nn.Conv2d(self.fc_hidden_dim, self.fc_hidden_dim, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(self.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fc_hidden_dim, self.fc_hidden_dim, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(self.fc_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.Caconv = nn.Sequential(
            nn.Conv2d(self.fc_hidden_dim, self.fc_hidden_dim, kernel_size=(5, 2), padding=(2, 0)),
            nn.BatchNorm2d(self.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fc_hidden_dim, self.fc_hidden_dim, kernel_size=(5, 2), padding=(2, 0)),
            nn.BatchNorm2d(self.fc_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_1x1 = nn.Conv2d(2*self.fc_hidden_dim, 
                                  self.fc_hidden_dim,
                                  kernel_size= 1,
                                  stride = 1,
                                  bias=False)
        # self.seq1 = nn.Sequential(
        #     nn.Conv2d(self.fc_hidden_dim, self.fc_hidden_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.fc_hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # self.seq2 = nn.Sequential(
        #     nn.Linear(self.stage, self.fc_hidden_dim//4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.fc_hidden_dim//4, self.stage),
        #     nn.Sigmoid()
        # )

    def forward(self, x, batch_features): #x is a List of tensor with shape of[4608, 64, 36, 1]
        batch_size = batch_features[0].shape[0]
        pooled_feat = []
        for i in range(self.stage):
            pooled_feat.append(self.pool(self.cala_weight_conv[i](batch_features[i])).squeeze())
        cala_weight_feat = torch.cat(pooled_feat, dim=-1)
        weight = self.cala_weight_fc(cala_weight_feat).reshape(batch_size, self.stage, 1, 1, 1)

        output = torch.stack(x, dim=1) #[24, 3, 192, 64, 36]
        # weight = output.clone()
        # weight = self.seq1(weight).transpose(1, 3)
        # weight = self.pool(weight).transpose(1, 3) #[4608, 1, 1, 3]
        # weight = self.seq2(weight)

        fusion_feat = torch.mul(output, weight).permute(0,2,3,4,1).flatten(0,1)
        return_feat = torch.cat([self.Caconv(fusion_feat), self.Caconv_1(fusion_feat.clone())],
                                dim = 1) #[4608, 64, 36, 1]
        
        return self.conv_1x1(return_feat) 
