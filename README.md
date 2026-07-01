# TDFSI-RSCD
“Temporal Difference Learning via Frequency-Spatial Interaction for Remote Sensing Change Detection” (TDFSI) 的官方实现代码

# TDFSI: 基于频空交互的时序差异学习遥感变化检测

本仓库提供了用于遥感变化检测 (RSCD) 的 TDFSI 框架的官方实现代码[cite: 4]。

## 项目简介 (Overview)
在复杂的遥感场景中，非变化区域经常由于光照变化、季节交替和背景杂乱而表现出特征差异[cite: 4]。这些因素很容易导致伪变化和模糊的边界[cite: 4]。为了解决这个问题，我们提出了基于频空交互的时序差异学习（TDFSI）框架[cite: 4]。我们的方法在空间域增强了时序差异表示，并在频域中对其进行了进一步的细化，从而实现更鲁棒、更精确的变化检测[cite: 4]。

## 框架结构 (Framework Architecture)
提出的 TDFSI 框架主要由三个核心组件构成[cite: 4]：

* **时序差异增强模块 (TDEM)**：旨在强化双时相特征中对变化敏感的差异线索，同时抑制不稳定的背景响应[cite: 4]。
* **频域变化细化模块 (FCR)**：在频谱域中细化学习到的时序差异表示，以增强全局一致性、减少伪变化噪声并锐化目标边界[cite: 4]。
* **变化解码器 (Change Decoder)**：利用基于 UPerNet 的结构逐步融合细化后的多级特征，并生成最终的精确变化掩膜[cite: 4]。

## 性能表现 (Performance)
TDFSI 模型在两个流行的基准数据集（LEVIR-CD 和 SYSU-CD）上进行了评估[cite: 4]。大量的实验证明了该方法的有效性[cite: 4]。

* 在以建筑为主的 LEVIR-CD 数据集上，TDFSI 取得了 **92.13%** 的 F1-score 和 **84.70%** 的 IoU[cite: 4]。
* 在 SYSU-CD 数据集上，即使面临明显的季节和光照变化等复杂场景，该方法依然展示了卓越的性能和极高的鲁棒性[cite: 4]。
