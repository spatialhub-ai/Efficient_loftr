import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        block_dims = config['backbone']['block_dims']
        self.W = int(self.config['fine_window_size'])
        self.fine_d_model = block_dims[0]

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1):
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2+feat_c)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1+x2)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        return x1
    
    def forward(self, feat_c0, feat_c1, data):
        W = self.W
        
        h0c, w0c = data['hw0_c'][0], data['hw0_c'][1]
        stride = int(self.config['resolution'][0] // self.config['resolution'][1])

        data.update({'W': W})

        feat_c_combined = torch.cat([feat_c0, feat_c1], 0).contiguous()
        feat_c = feat_c_combined.view(-1, h0c, w0c, feat_c0.shape[-1]).permute(0, 3, 1, 2) # 1/8 feat
        
        x2 = data['feats_x2'] # 1/4 feat
        x1 = data['feats_x1'] # 1/2 feat
        del data['feats_x2'], data['feats_x1']

        # 1. fine feature extraction
        x1 = self.inter_fpn(feat_c, x2, x1)                    
        feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

        # 2. unfold(crop) all local windows
        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
        C, L = feat_f0.shape[1] // (W**2), feat_f0.shape[2]
        feat_f0 = feat_f0.view(-1, C, W**2, L).permute(0, 3, 2, 1)

        feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
        C_pad, L_pad = feat_f1.shape[1] // ((W+2)**2), feat_f1.shape[2]
        feat_f1 = feat_f1.view(-1, C_pad, (W+2)**2, L_pad).permute(0, 3, 2, 1)

        # 3. ONNX-Safe Match Flattened Indexing Selection
        b_ids = data['b_ids']
        i_ids = data['i_ids']
        j_ids = data['j_ids']

        # Flatten the batch and window locations into a single global dimension to simplify gather operations
        feat_f0 = feat_f0.reshape(-1, W**2, self.fine_d_model)
        feat_f1 = feat_f1.reshape(-1, (W+2)**2, self.fine_d_model)

        flat_indices_i = b_ids * L + i_ids
        flat_indices_j = b_ids * L_pad + j_ids

        return feat_f0[flat_indices_i], feat_f1[flat_indices_j]
