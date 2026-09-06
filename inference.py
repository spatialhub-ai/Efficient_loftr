import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter

def crop_to_same_size(img0, img1):
    """
    Center-crop both images so they become the same size.
    The larger image is cropped down to match the smaller one.
    """

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    target_h = min(h0, h1)
    target_w = min(w0, w1)

    def center_crop(img, th, tw):
        h, w = img.shape[:2]

        start_x = (w - tw) // 2
        start_y = (h - th) // 2

        return img[start_y:start_y + th, start_x:start_x + tw]

    img0_cropped = center_crop(img0, target_h, target_w)
    img1_cropped = center_crop(img1, target_h, target_w)

    return img0_cropped, img1_cropped

def visualize_matches(img0_gray, img1_gray, mkpts0, mkpts1, mconf=None, conf_thresh=0.5, max_side=800):

    img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

    # Scale images so largest dimension <= max_side
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    scale0 = min(1.0, max_side / max(h0, w0))
    scale1 = min(1.0, max_side / max(h1, w1))

    img0 = cv2.resize(
        img0,
        (int(w0 * scale0), int(h0 * scale0)),
        interpolation=cv2.INTER_AREA,
    )

    img1 = cv2.resize(
        img1,
        (int(w1 * scale1), int(h1 * scale1)),
        interpolation=cv2.INTER_AREA,
    )

    mkpts0 = mkpts0 * scale0
    mkpts1 = mkpts1 * scale1

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    H = max(h0, h1)

    canvas0 = np.zeros((H, w0, 3), dtype=np.uint8)
    canvas1 = np.zeros((H, w1, 3), dtype=np.uint8)

    canvas0[:h0] = img0
    canvas1[:h1] = img1

    vis = np.hstack([canvas0, canvas1])

    for i in range(len(mkpts0)):

        if mconf is not None and mconf[i] < conf_thresh:
            continue

        x0, y0 = mkpts0[i]
        x1, y1 = mkpts1[i]

        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)

        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.circle(vis, (x0, y0), 3, color, -1)
        cv2.circle(vis, (x1 + w0, y1), 3, color, -1)

        cv2.line(vis, (x0, y0), (x1 + w0, y1), color, 1, cv2.LINE_AA,)

    cv2.imshow("LoFTR Matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def infer():
    # Initialize the matcher with default settings
    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)

    # Load pretrained weights
    matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt", weights_only=False)['state_dict'])
    matcher = reparameter(matcher)  # Essential for good performance
    matcher = matcher.eval().cuda()


    # Load and preprocess images
    img0_raw = cv2.imread("data/0015/images/29307281_d7872975e2_o.jpg", cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread("data/0015/images/50646217_c352086389_o.jpg", cv2.IMREAD_GRAYSCALE)

    img0_raw, img1_raw = crop_to_same_size(img0_raw, img1_raw)

    # Resize images to be divisible by 32
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

    # Convert to tensors
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

    # Inference
    with torch.no_grad():
        mkpts0, mkpts1, mconf = matcher(img0, img1)
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()
        mconf = mconf.cpu().numpy()

        idx = np.argsort(-mconf)[:50]

        visualize_matches(img0_raw, img1_raw, mkpts0[idx], mkpts1[idx], mconf[idx], conf_thresh=0.5, max_side=800,)

if __name__ == "__main__":
    infer()

