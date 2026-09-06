import torch
import torch.nn as nn

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher

class LoFTR(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler

        # Modules
        self.backbone = build_backbone(config)            
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)

    def forward(self, image0, image1):
        """ 
        Update:
            'image0': (torch.Tensor): (N, 1, H, W)
            'image1': (torch.Tensor): (N, 1, H, W)
        """
        data = {}

        # 1. Local Feature CNN
        bs = image0.shape[0]
        data.update({
            'bs': bs,
            'hw0_i': image0.shape[2:], 
            'hw1_i': image1.shape[2:]
        })

        # backbone((2N, 1, H, W)) -> {feats_c: (2N, C, H/8, W/8), feats_x2: (2N, C2, H/4, W/4), feats_x1: (2N, C1, H/2, W/2)}
        ret_dict = self.backbone(torch.cat([image0, image1], dim=0))
        
        # feat_c0: (N, C, H/8, W/8), feat_c1: (N, C, H/8, W/8)
        feat_c0 = ret_dict['feats_c'][:data['bs']]
        feat_c1 = ret_dict['feats_c'][data['bs']:]
        
        data.update({'feats_x2': ret_dict['feats_x2'], 'feats_x1': ret_dict['feats_x1'],})

        mul = int(self.config['resolution'][0] // self.config['resolution'][1])

        # hw0_c: (H/8, W/8), hw1_c: (H/8, W/8), hw0_f: (H, W), hw1_f: (H, W)
        data.update({
            'hw0_c': feat_c0.shape[2:],
            'hw1_c': feat_c1.shape[2:],
            'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
            'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
        })

        # 2. coarse-level loftr module
        # Refine coarse features while preserving channels and spatial resolution.
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)

        # feat_c0: (N, C, H/8, W/8) -> (N, H/8*W/8, C), feat_c1: (N, C, H/8, W/8) -> (N, H/8*W/8, C)
        feat_c0 = feat_c0.permute(0, 2, 3, 1).flatten(1, 2)
        feat_c1 = feat_c1.permute(0, 2, 3, 1).flatten(1, 2)
            
        # prevent fp16 overflow during mixed precision training
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=None, mask_c1=None)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, data)
        
        del feat_c0, feat_c1

        # 5. match fine-level            
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

