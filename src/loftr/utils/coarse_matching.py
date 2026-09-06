import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_border(m, b: int, v):
    """ Mask border regions in the 4D matching volume [N, H0, W0, H1, W1] to suppress unreliable matches near image edges caused by padding 
        or limited receptive field effects.
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    # Explicit inplace assignments that translate cleanly to standard ONNX slice/scatter
    m[:, :b, :, :, :] = v
    m[:, :, :b, :, :] = v
    m[:, :, :, :b, :] = v
    m[:, :, :, :, :b] = v
    m[:, -b:, :, :, :] = v
    m[:, :, -b:, :, :] = v
    m[:, :, :, -b:, :] = v
    m[:, :, :, :, -b:] = v
    return m


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']
        self.skip_softmax = config['skip_softmax']
        self.fp16matmul = config['fp16matmul']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        if self.fp16matmul:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            del feat_c0, feat_c1
        else:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            del feat_c0, feat_c1

        if not self.skip_softmax:
            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': sim_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(sim_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        h0c, w0c, h1c, w1c = data['hw0_c'][0], data['hw0_c'][1], data['hw1_c'][0], data['hw1_c'][1]

        # 1. confidence thresholding
        mask = (conf_matrix > self.thr).contiguous()

        mask = mask.view(-1, h0c, w0c, h1c, w1c)
        mask_border(mask, self.border_rm, False)
        mask = mask.view(-1, h0c * w0c, h1c * w1c)
            
        # 2. mutual nearest
        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_int = mask.to(torch.int8)
        mask_v, all_j_ids = mask_int.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = 8.0 # backbone downsampling factor
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        
        # Compute matching keypoints coordinates
        w0c_tensor = torch.empty((), device=conf_matrix.device, dtype=i_ids.dtype).fill_(w0c)
        w1c_tensor = torch.empty((), device=conf_matrix.device, dtype=j_ids.dtype).fill_(w1c)

        mkpts0_c = torch.stack([i_ids % w0c_tensor, torch.div(i_ids, w0c_tensor, rounding_mode='trunc')], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % w1c_tensor, torch.div(j_ids, w1c_tensor, rounding_mode='trunc')], dim=1) * scale1
        
        valid_mask = (mconf != 0)
        m_bids = b_ids[valid_mask]        
        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'm_bids': m_bids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[valid_mask],
            'mkpts1_c': mkpts1_c[valid_mask],
            'mconf': mconf[valid_mask]
        })

        return coarse_matches
