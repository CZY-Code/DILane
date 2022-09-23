import torch
import torch.nn.functional as F
import copy
import torch.nn as nn


class DynamicConv(nn.Module): #
    def __init__(self, n, feat_size, inplanes, early_return):
        super().__init__()
        self.early_return = early_return
        self.queryNum = n
        self.hidden_dim = inplanes #是不是太大了
        self.dim_dynamic = inplanes*2
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic #49152
        self.dynamic_layer = nn.Linear(self.hidden_dim * feat_size, self.num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace = True)

        num_output = self.hidden_dim * feat_size
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_feature, roi_feature): ##[24, 192, 64] [b, 192, 36, 64]
        bs = roi_feature.shape[0]
        roi_feature = roi_feature.flatten(start_dim=0, end_dim=1) #[8*7, 12, 64] 此处第1维度的大小还可以改
        pro_feature = pro_feature.flatten(start_dim=0, end_dim=1).unsqueeze(0) #[1, 4608, 64]

        parameters = self.dynamic_layer(pro_feature) #[1, 8*7, 64*16*2]

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(roi_feature, param1) #bmm输入维度只能是三维
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features) # [4608, 36, 64]
        
        if self.early_return:
            return features

        #还有一个线性层
        features = features.flatten(1)
        features = self.out_layer(features) #将有HW大小的roi_feat转换为没有HW=1和pro_feat一样大的特征
        features = self.norm3(features)
        features = features.view(bs, self.queryNum, -1)
        return features


class DynamicHead(nn.Module): 
    '''对每张图输入的一个 7*queries*C大小的Tensor，先计算得到不同queries的权重，然后模仿SparseRCNN对不同proposal分别进行回归预测
        将queries_embed看为proposal_bbox'''
    def __init__(self, n, feat_size, inplanes=64, early_return=False):
        super().__init__()
        self.inplanes = inplanes
        self.feat_size = feat_size
        self.queryNum = n
        self.proposalFeat = nn.Embedding(n, inplanes)
        self.self_attn = nn.MultiheadAttention(embed_dim=inplanes, num_heads=8, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(inplanes)

        self.inst_interact = DynamicConv(n, feat_size, inplanes, early_return)
        self.linear1 = nn.Linear(inplanes, 128)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128, inplanes)
        self.norm2 = nn.LayerNorm(inplanes)
        self.norm3 = nn.LayerNorm(inplanes)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.activation = F.relu

        reg_module = list()
        for _ in range(1):
            reg_module.append(nn.Linear(inplanes, inplanes, False))
            reg_module.append(nn.LayerNorm(inplanes))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

    def forward(self, proposalFeat, features): #[24, 192, 64] #[b, 192, 36, 64]
        #self_att
        
        # proposalFeat = proposalFeat.transpose(0, 1)
        #proposalFeat = self.proposalFeat.weight.unsqueeze(1).repeat(1, bs, 1)
        # proposalFeat2 = self.self_attn(proposalFeat, proposalFeat, value=proposalFeat)[0]
        # proposalFeat = proposalFeat + self.dropout1(proposalFeat2)
        # proposalFeat = self.norm1(proposalFeat).transpose(0, 1) #[24, 192, 64]

        #features = features.view(self.queryNum, bs, self.feat_size, self.inplanes)
        features2 = self.inst_interact(proposalFeat, features) #[192, 8, 64] [b, 192, feat_size, 64]
        return features2

        proposalFeat = proposalFeat.transpose(0, 1) #[8, 7, 64]
        features = proposalFeat + self.dropout2(features2) 
        obj_features = self.norm2(features)
        #线性层
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features) #[B, N, 64]

        reg_feature = obj_features.clone() #不共享内存但要叠加梯度
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature) #作为query_pos反馈回去

        return reg_feature, obj_features
