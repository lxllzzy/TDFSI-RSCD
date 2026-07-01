import timm
from lib.pvt_v2 import pvt_v2_b4
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.upernet_light import UperNetHead, EinFFT, DiffFusionBlock, MultiScaleDiffExtractor



'''
backbone: PVT_v2_b4
'''


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=23, stride=1, padding=11) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

class Network(nn.Module):
    def __init__(self, channels=128, device: str = "cuda"):
        super(Network, self).__init__()
        self.shared_encoder = pvt_v2_b4()
        pretrained_dict = torch.load('/data/lxl/pre_train_pth/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )
        self.diff_extractor = MultiScaleDiffExtractor(ch_per_stage=[512, 320, 128, 64])

        self.device = torch.device(device)
        pos_weight = torch.tensor(5.0, device=self.device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.changefft1 = EinFFT(dim=64)  # stage1 通道数
        self.changefft2 = EinFFT(dim=128)  # stage2 通道数
        self.changefft3 = EinFFT(dim=320)  # stage3 通道数
        self.changefft4 = EinFFT(dim=512)  # stage4 通道数
        config = type('', (), {})()
        config.pool_scales = (1, 2, 3, 6)
        config.hidden_size = 256
        config.num_labels = 1
        self.decode_head = UperNetHead(config, in_channels=[64, 128, 320, 512])


    def dice(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum() + eps
        den = pred.sum() + target.sum() + eps
        return 1 - num / den

    def forward(self, data):
        imgs1 = data["image1_m"]
        imgs2 = data["image2_m"]

        feats1 = self.shared_encoder(imgs1)
        feats2 = self.shared_encoder(imgs2)


        diff_feats = self.diff_extractor(feats1, feats2)
        x4, x3, x2, x1 = diff_feats

        #changeFFT模块
        dF1 = self.changefft1(x1)  # ΔF1
        dF2 = self.changefft2(x2)  # ΔF2
        dF3 = self.changefft3(x3)  # ΔF3
        dF4 = self.changefft4(x4)  # ΔF4
        output = self.decode_head([dF1, dF2, dF3, dF4])
        # output = self.decode_head([x1, x2, x3, x4])
        pred = F.interpolate(output, size=imgs1.shape[-2:], mode="bilinear", align_corners=False)

        # --- 训练阶段 ---
        mask = data["mask"].float()
        prob = torch.sigmoid(pred)


        if not self.training:
            prob = torch.sigmoid(pred)
            return {
                "pred_mask": prob,
                "vis": {"pred": prob}
            }
        loss = self.bce(pred, mask)  # labels: (B,H,W), float {0.0,1.0}

        return {
            "loss": loss,
            "pred_mask": prob.detach(),
            "vis": {"pred": prob.detach()},
            "loss_dict": {
                "init": 0,
                "final": 0,
                "total": loss.item()
            }
        }


