import os, argparse, yaml
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "7"

import torch
from torch.utils import data
from tqdm import tqdm
from datetime import datetime
from dataset.changedataset import ChangeTrainDataset
from utils import io, pt_utils, metric_utils
from PIL import Image
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 直接导入 Network_Fre
from lib.Network_cd import Network
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


# ============ Step Decay (备用) ============
class StepDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, decay_rate=0.98, decay_epoch=10, last_epoch=-1):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        super(StepDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.decay_rate ** (self.last_epoch // self.decay_epoch)
        return [self.init_lr * factor for _ in self.base_lrs]


# ============ 训练函数 ============
def train(model, optimizer, scheduler, train_loader, val_loader, accelerator, num_epochs, grad_acc_step=1):

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/exp_acc_{time_str}"
    if accelerator.is_main_process:
        print(f"当前日志目录: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    best_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            disable=not accelerator.is_main_process,
                            desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in progress_bar:
            data_batch = pt_utils.to_device(batch["data"], accelerator.device)

            with accelerator.autocast():
                outputs = model(data=data_batch)
                loss = outputs["loss"] / grad_acc_step
                loss_info = outputs.get("loss_dict", {})

            accelerator.backward(loss)

            if (step + 1) % grad_acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if writer and accelerator.is_main_process:
                    true_step = epoch * (len(train_loader) // grad_acc_step) + (step + 1) // grad_acc_step
                    writer.add_scalar("Loss/total", loss.item() * grad_acc_step, true_step)
                    for k in ["init", "final"]:
                        if k in loss_info:
                            writer.add_scalar(f"Loss/{k}", loss_info[k], true_step)

            # 打印
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'init': f'{loss_info.get("init", 0):.4f}',
                    'final': f'{loss_info.get("final", 0):.4f}',
                    'lr': f'{current_lr:.8f}'
                })

        # ===== 每个 epoch 做一次验证 =====
        if val_loader:
            scores = validate(model, val_loader, accelerator, epoch, log_dir)
            if writer and accelerator.is_main_process:
                writer.add_scalar("Metric/IoU", scores["iou"], epoch)
                writer.add_scalar("Metric/F1", scores["f1"], epoch)

            # 保存最佳模型
            if scores["iou"] > best_iou and epoch > 5:
                best_iou = scores["iou"]
                if accelerator.is_main_process:
                    io.save_weight("weights/3_fft_best_acc_SYSU_FDconv.pth", accelerator.unwrap_model(model))
                    print(f"[Epoch {epoch+1}] Save best model (IoU={best_iou:.4f})")

    if writer:
        writer.close()
    accelerator.print("Training finished.")


# ============ 验证函数 ============
@torch.no_grad()
def validate(model, val_loader, accelerator, epoch, log_dir):
    model.eval()
    metric_buf = metric_utils.MetricBuffer()

    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "val"))

    num_batches = len(val_loader)
    for idx, batch in enumerate(val_loader):
        data_batch = pt_utils.to_device(batch["data"], accelerator.device)
        outputs = model(data=data_batch)

        pred_mask = (outputs["pred_mask"] > 0.5).int()
        gt_mask = data_batch["mask"].int()

        metric_buf.update(pred_mask.cpu(), gt_mask.cpu())

        if writer:
            B = pred_mask.shape[0]
            vis_N = min(B, 4)
            canvases = []
            for b in range(vis_N):
                gt_np = gt_mask[b].squeeze().cpu().numpy().astype(bool)
                pred_np = pred_mask[b, 0].squeeze().cpu().numpy().astype(bool)
                H, W = gt_np.shape
                canvas = np.zeros((H, W, 3), dtype=np.uint8)
                canvas[gt_np & pred_np] = [0, 255, 0]
                canvas[gt_np & ~pred_np] = [0, 0, 255]
                canvas[~gt_np & pred_np] = [255, 0, 0]
                canv_t = torch.from_numpy(canvas).permute(2, 0, 1)
                canvases.append(canv_t)
            grid = torchvision.utils.make_grid(torch.stack(canvases), nrow=vis_N)
            writer.add_image("Val/overlay", grid.float() / 255.0, epoch * num_batches + idx)

    if writer:
        writer.close()

    scores = metric_buf.compute()
    accelerator.print(f"[Val] IoU={scores['iou']:.4f}  F1={scores['f1']:.4f}")
    return scores


# ============ 配置文件解析 ============
def parse_cfg():
    parser = argparse.ArgumentParser("Change detection training")
    parser.add_argument("--config", default="./train.yaml", type=str)
    parser.add_argument("--data-root", default="/data/lxl/data/SYSU-CD256/", type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg["data_root"] = args.data_root
    return cfg


# ============ 主函数 ============
def main():
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(ddp_kwargs=ddp_kwargs)  # 允许未用参数，避免“没梯度”报错
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    cfg = parse_cfg()

    # Dataset
    tr_dataset = ChangeTrainDataset(
        root_dir=os.path.join(cfg["data_root"], "train"),
        shape=cfg["train"]["shape"]
    )
    val_loader = None
    val_root = os.path.join(cfg["data_root"], "val")
    print(val_root)
    if os.path.isdir(val_root):
        val_dataset = ChangeTrainDataset(
            root_dir=val_root,
            shape=cfg["train"]["shape"]
        )
        val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    tr_loader = data.DataLoader(
        tr_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 8),
        pin_memory=True,
        drop_last=True
    )
    print(len(tr_loader))
    print(len(val_loader))

    # 直接加载 Network_Fre
    model = Network()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"] * len(tr_loader),
        eta_min=cfg["train"].get("min_lr", 1.0e-6)
    )

    # accelerator 处理
    model, optimizer, tr_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, tr_loader, val_loader, scheduler
    )

    # Training
    train(model,
          optimizer,
          scheduler,
          tr_loader,
          val_loader,
          accelerator,
          num_epochs=cfg["train"]["epochs"],
          grad_acc_step=cfg["train"].get("grad_acc_step", 1))


if __name__ == "__main__":
    main()
