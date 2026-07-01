import os, argparse, yaml
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from torch.utils import data
from tqdm import tqdm
from dataset.changedataset import ChangeTrainDataset
from utils import io, pt_utils, metric_utils
from PIL import Image
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lib.Network_cd import Network
from lib.compare.FC_EF import Unet
from lib.compare.FC_Siam_Conc import SiamUnet_conc
from lib.compare.FC_Siam_Diff import SiamUnet_diff
from lib.compare.SNUNet import SNUNet_ECAM


# ============ 测试函数 ============
@torch.no_grad()
def test(model, test_loader, accelerator, save_dir=None, save_metrics=True):
    model.eval()
    metric_buf = metric_utils.MetricBuffer()

    progress_bar = tqdm(test_loader, disable=not accelerator.is_main_process, desc="Testing")
    for idx, batch in enumerate(progress_bar):
        data_batch = pt_utils.to_device(batch["data"], accelerator.device)
        outputs = model(data=data_batch)

        pred_mask = (outputs["pred_mask"] > 0.5).int()
        gt_mask = data_batch["mask"].int()

        # 更新指标
        metric_buf.update(pred_mask.cpu(), gt_mask.cpu())

        # 保存预测结果
        if save_dir and accelerator.is_main_process:
            B = pred_mask.shape[0]
            for b in range(B):
                pred_np = pred_mask[b, 0].cpu().numpy().astype(np.uint8) * 255
                img = Image.fromarray(pred_np)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{idx:05d}_{b}.png")
                img.save(save_path)

    # 计算指标
    scores = metric_buf.compute()
    accelerator.print("[Test Results]")
    for k, v in scores.items():
        accelerator.print(f"  {k}: {v:.4f}")

    # 保存指标
    if save_metrics and accelerator.is_main_process:
        import json
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        accelerator.print(f"✅ Metrics saved to {metrics_path}")

    return scores


# ============ 配置文件解析 ============
def parse_cfg():
    parser = argparse.ArgumentParser("Change detection testing")
    parser.add_argument("--config", default="./train.yaml", type=str)
    parser.add_argument("--data-root", default="/data/lxl/data/SYSU-CD256", type=str)
    parser.add_argument("--weight", default="/data/lxl/FSEL/FSEL_ECCV_2024/weights/3_fft_best_acc_SYSU_FDconv.pth", type=str)
    parser.add_argument("--save-dir", default="predictions_vis", type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg["data_root"] = args.data_root
    cfg["weight"] = args.weight
    cfg["save_dir"] = args.save_dir
    return cfg


# ============ 主函数 ============
def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    cfg = parse_cfg()

    # Dataset
    test_root = os.path.join(cfg["data_root"], "test")
    test_dataset = ChangeTrainDataset(
        root_dir=test_root,
        shape=cfg["train"]["shape"],
        is_train=False
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 8),
        pin_memory=True
    )
    print(len(test_loader))
    # 模型
    # model = SNUNet_ECAM(in_ch=3, out_ch=1)
    model = Network()
    io.load_weight(cfg["weight"], model)

    # accelerator 处理
    model, test_loader = accelerator.prepare(model, test_loader)

    # 测试
    test(model, test_loader, accelerator, save_dir=cfg["save_dir"])


if __name__ == "__main__":
    main()
