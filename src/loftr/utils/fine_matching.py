import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.geometry.grid import create_meshgrid

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, _, C = feat_0.shape
        W = int(self.config['fine_window_size'])
        WW = int(W**2)

        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # compute pixel-level confidence matrix
        feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
        feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
        feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
        
        conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
        conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].contiguous().view(M, self.WW, self.WW)

        # compute pixel-level absolute kpt coords
        self.get_fine_ds_match(softmax_matrix_f, data)

        # generate seconde-stage 3x3 grid
        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long)
        idx_r_iids, idx_r_jids = torch.div(idx_r, W, rounding_mode="trunc"), idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.view(-1), idx_l.view(-1), idx_r_iids.view(-1), idx_r_jids.view(-1)
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long) # [1, 3, 3, 2]

        m_ids = m_ids[...,None,None].expand(-1, 3, 3)
        idx_l = idx_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.view(-1, self.WW, self.W+2, self.W+2)

        # Flatten multi-dimensional gather for ONNX compatibility
        flat_indices = (m_ids * WW * (W+2) * (W+2)) + (idx_l * (W+2) * (W+2)) + (idx_r_iids * (W+2)) + idx_r_jids
        conf_matrix_ff = conf_matrix_ff.view(-1)
        conf_matrix_ff = conf_matrix_ff[flat_indices.view(-1)]
        
        conf_matrix_ff = conf_matrix_ff.view(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, dim=-1)
        heatmap = conf_matrix_ff.view(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]

        scale1 = data.get('scale1', torch.ones((), device=feat_0.device))
        scale1 = scale * scale1
        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(coords_normalized, data, scale1)

    def get_fine_match_local(self, coords_normed, data, scale1):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3.0 / 2.0) * scale1)

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        M = conf_matrix.shape[0]

        conf_matrix = conf_matrix.view(M, -1)
        val, idx = torch.max(conf_matrix, dim = -1)
        idx = idx.view(-1, 1)

        idx_l, idx_r = torch.div(idx, self.WW, rounding_mode="trunc"), idx % self.WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        grid = create_meshgrid(self.W, self.W, False, conf_matrix.device, dtype=conf_matrix.dtype) - self.W // 2 + 0.5 # kornia >= 0.5.1
        grid = grid.reshape(1, -1, 2).expand(M, -1, -1)

        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = data.get('scale0', torch.ones((), device=conf_matrix.device))
        scale1 = data.get('scale1', torch.ones((), device=conf_matrix.device))

        mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0 * self.scale)).view(-1, 2)
        mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1 * self.scale)).view(-1, 2)
        
        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f
        })

