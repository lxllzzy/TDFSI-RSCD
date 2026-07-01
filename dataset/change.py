import numpy as np
from torch.utils.data import Dataset
import os
import imageio.v2 as imageio
from . import transforms
# import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch
import cv2
import json


def load_change_labels(label_file):
    """加载图像级变化标签"""
    if os.path.exists(label_file):
        labels = np.load(label_file, allow_pickle=True).item()
        if labels is None:
            print(f"Warning: {label_file} is empty.")
            return {}
        return labels
    print(f"Warning: Label file {label_file} not found.")
    return {}  # 如果文件不存在或为空，返回一个空字典


def _transform_normalize():
    """标准化转换"""
    return Compose([
        ToTensor(),
        # 使用遥感图像的统计值
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class RSChangeDataset(Dataset):
    """遥感变化检测数据集的基础类 - 直接从目录加载"""

    def __init__(
            self,
            data_dir=None,  # 数据集目录，如train_dir, val_dir或test_dir
            stage='train',  # 'train', 'val', 或 'test'
    ):
        super().__init__()

        self.data_dir = data_dir
        self.stage = stage

        # 两个时相的图像目录
        self.img_t1_dir = os.path.join(data_dir, 't1')
        self.img_t2_dir = os.path.join(data_dir, 't2')

        # 检查目录是否存在
        if not os.path.exists(self.img_t1_dir):
            raise ValueError(f"T1 directory not found: {self.img_t1_dir}")
        if not os.path.exists(self.img_t2_dir):
            raise ValueError(f"T2 directory not found: {self.img_t2_dir}")

        # 获取T1目录下的所有图像文件
        self.img_files = [f for f in os.listdir(self.img_t1_dir)
                          if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

        # 验证T2目录中是否有对应的文件
        valid_files = []
        for img_file in self.img_files:
            if os.path.exists(os.path.join(self.img_t2_dir, img_file)):
                valid_files.append(img_file)
            else:
                print(f"Warning: {img_file} exists in T1 but not in T2")

        self.img_files = valid_files

        # 加载图像级变化标签
        label_file = os.path.join(data_dir, f'change_types_{stage}.npy')
        path = label_file
        self.change_labels = load_change_labels(label_file)

        if self.change_labels is None:
            print(f"Warning: No change labels found at {label_file}")
            print("Using dummy labels (all zeros)")
            # 创建虚拟标签
            self.change_labels = {name: 0 for name in self.img_files}

        print(f"Loaded {len(self.img_files)} images from {data_dir}")

        # 统计有变化和无变化的图像数量
        if self.change_labels:
            change_count = sum(1 for name in self.img_files if self.change_labels.get(name, 0) == 1)
            no_change_count = sum(1 for name in self.img_files if self.change_labels.get(name, 0) == 0)
            print(f"Images with changes: {change_count}, without changes: {no_change_count}")

    def __len__(self):
        return len(self.img_files)



    def __getitem__(self, idx):
        img_name = self.img_files[idx]

        # 读取两个时相的图像
        img_t1_path = os.path.join(self.img_t1_dir, img_name)
        img_t2_path = os.path.join(self.img_t2_dir, img_name)

        image_t1 = np.asarray(imageio.imread(img_t1_path))
        image_t2 = np.asarray(imageio.imread(img_t2_path))

        # 确保图像是3通道RGB
        if len(image_t1.shape) == 2:
            image_t1 = np.stack([image_t1, image_t1, image_t1], axis=2)
        if len(image_t2.shape) == 2:
            image_t2 = np.stack([image_t2, image_t2, image_t2], axis=2)
        if image_t1.shape[2] > 3:
            image_t1 = image_t1[:, :, :3]
        if image_t2.shape[2] > 3:
            image_t2 = image_t2[:, :, :3]

        # 获取图像级变化标签
        # change_label = self.change_labels.get(img_name, 0)
        val = self.change_labels.get(img_name, 0)
        if isinstance(val, np.ndarray):
            change_label = int(val[0])  # 把 array([0.]) 转成 0
        else:
            change_label = int(val)

        # ====== 修改点 1: 支持真实像素级标签 ======
        label_dir = os.path.join(self.data_dir, "label")
        label_path = os.path.join(label_dir, img_name)
        if os.path.exists(label_path):
            dummy_label = np.asarray(imageio.imread(label_path))
        else:
            dummy_label = np.zeros((image_t1.shape[0], image_t1.shape[1]), dtype=np.uint8)
        # =====================================

        return img_name, image_t1, image_t2, dummy_label, change_label


class RSChangeDetectionDataset(RSChangeDataset):
    """遥感变化检测数据集的增强版本 - 直接从目录加载"""

    def __init__(self,
                 data_dir=None,
                 stage='train',
                 resize_range=[256, 256],
                 rescale_range=[1.0, 1.0],
                 crop_size=256,
                 img_fliplr=True,
                 img_flipud=True,
                 rotate_range=[-45, 45],
                 ignore_index=255,
                 num_classes=2,  # 二分类：变化/未变化
                 aug=False
                 ):

        super().__init__(data_dir, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.img_flipud = img_flipud
        self.rotate_range = rotate_range
        self.num_classes = num_classes

        # 颜色增强
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.normalize = _transform_normalize()

    def __transforms(self, image_t1, image_t2):
        """对两个时相图像应用相同的变换"""
        if self.aug:
            # 调整大小
            if self.resize_range:
                h, w = self.resize_range
                image_t1 = cv2.resize(image_t1, (w, h), interpolation=cv2.INTER_LINEAR)
                image_t2 = cv2.resize(image_t2, (w, h), interpolation=cv2.INTER_LINEAR)

            # ====== 修改点 2: 同步缩放 ======
            if self.rescale_range:
                scale = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
                image_t1 = cv2.resize(image_t1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                image_t2 = cv2.resize(image_t2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # =================================

            # 随机水平翻转
            if self.img_fliplr and np.random.rand() > 0.5:
                image_t1 = np.fliplr(image_t1)
                image_t2 = np.fliplr(image_t2)

            # 随机垂直翻转
            if self.img_flipud and np.random.rand() > 0.5:
                image_t1 = np.flipud(image_t1)
                image_t2 = np.flipud(image_t2)

            # ====== 修改点 3: 改进旋转边界填充 ======
            if self.rotate_range and np.random.rand() > 0.5:
                angle = np.random.uniform(self.rotate_range[0], self.rotate_range[1])
                h, w = image_t1.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # 计算平均像素值作为边界填充值
                mean_val_t1 = tuple(map(int, image_t1.mean(axis=(0, 1))))
                mean_val_t2 = tuple(map(int, image_t2.mean(axis=(0, 1))))

                image_t1 = cv2.warpAffine(image_t1, M, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=mean_val_t1)
                image_t2 = cv2.warpAffine(image_t2, M, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=mean_val_t2)

            # 随机裁剪
            if self.crop_size:
                # 对两个时相使用相同的裁剪区域
                h, w = image_t1.shape[:2]
                crop_h, crop_w = self.crop_size, self.crop_size

                # 确保裁剪区域不超过图像边界
                if h > crop_h and w > crop_w:
                    start_h = np.random.randint(0, h - crop_h + 1)
                    start_w = np.random.randint(0, w - crop_w + 1)

                    image_t1 = image_t1[start_h:start_h + crop_h, start_w:start_w + crop_w]
                    image_t2 = image_t2[start_h:start_h + crop_h, start_w:start_w + crop_w]
                else:
                    # 如果图像太小，则调整大小
                    image_t1 = cv2.resize(image_t1, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                    image_t2 = cv2.resize(image_t2, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            # 颜色增强（独立应用于每个时相）
            image_t1 = self.color_jittor(image_t1)
            image_t2 = self.color_jittor(image_t2)

        # 标准化
        image_t1 = transforms.normalize_img(image_t1)
        image_t2 = transforms.normalize_img(image_t2)

        # 转换为CHW格式
        image_t1 = np.transpose(image_t1, (2, 0, 1)).astype(np.float32)
        image_t2 = np.transpose(image_t2, (2, 0, 1)).astype(np.float32)

        return image_t1, image_t2

    def __getitem__(self, idx):

        img_name, image_t1, image_t2, dummy_label, change_label = super().__getitem__(idx)

        # 检查是否成功加载图像
        if image_t1 is None or image_t2 is None:
            print(f"Warning: Failed to load images for index {idx}")
            return None  # 或者返回默认值

        # 检查是否加载了有效的标签
        if change_label is None:
            print(f"Warning: No change label found for {img_name}, using default 0")
            change_label = 0  # 默认标签

        # 应用数据增强
        image_t1, image_t2 = self.__transforms(image_t1, image_t2)

        # 转换为PyTorch张量
        image_t1 = torch.from_numpy(image_t1)
        image_t2 = torch.from_numpy(image_t2)
        dummy_label = torch.from_numpy(dummy_label).long()
        change_label = torch.tensor(change_label).long()

        return {
            "data" : {
                "image1_m": image_t1,  # (3,H,W)
                "image2_m": image_t2,
                "mask": dummy_label,  # 可选
                "img_label": change_label
            },
            "meta": {
                "name": img_name,  # 字符串
            }

        }


class RSMultiClassChangeDataset(RSChangeDetectionDataset):
    """多类别变化检测数据集 - 直接从目录加载"""

    def __init__(self, num_classes=2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    train_dir = r"D:\project\datasets\LEVIR-CD+256\train"
    val_dir = r"D:\project\datasets\LEVIR-CD+256\val"
    ldataset = RSChangeDetectionDataset(
        data_dir=train_dir,
        stage="train",
        aug=False,
        crop_size=256
    )

    dataset = RSChangeDetectionDataset(
        data_dir=val_dir,  # 直接指定验证集目录
        stage='val',
        aug=False,
        ignore_index=255,
        num_classes=2
    )

    print("Dataset length:", len(dataset))

    # ====== 验证图像级标签统计 ======
    total_samples = len(dataset)
    change_label_counts = {0: 0, 1: 0}  # 统计0和1的数量

    # 遍历所有样本，统计change_label
    print("\n=== 开始统计图像级标签 ===")
    for i in range(total_samples):
        img_name, img_t1, img_t2, dummy_label, change_label = dataset[i]

        # 记录标签值
        change_label_counts[change_label] = change_label_counts.get(change_label, 0) + 1

        # 打印前10个样本的详细信息
        if i < 10:
            print(f"\nSample {i}:")
            print(f"  Image name: {img_name}")
            print(f"  Change label: {change_label}")
            print(f"  T1 text top-k: {text_t1}")
            print(f"  T2 text top-k: {text_t2}")

    # ====== 打印统计结果 ======
    print("\n=== 图像级标签统计结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"标签为0的样本数: {change_label_counts[0]}")
    print(f"标签为1的样本数: {change_label_counts[1]}")
    print(f"标签为0的比例: {change_label_counts[0] / total_samples:.4f}")
    print(f"标签为1的比例: {change_label_counts[1] / total_samples:.4f}")

    # 检查是否所有标签都是0
    if change_label_counts[1] == 0:
        print("\n🚨 警告: 验证集中所有样本的图像级标签都是0！")
    else:
        print(f"\n✅ 验证集中包含 {change_label_counts[1]} 个标签为1的样本")



