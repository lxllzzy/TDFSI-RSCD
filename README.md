🔗 TDFSI
TDFSI: Temporal Difference Learning via Frequency-Spatial Interaction for Remote Sensing Change Detection

Xianglong Liu, Fang Liu*, Jia Liu, Xu Tang, and Liang Xiao

Nanjing University of Science and Technology; Xidian University

🔭 Overview
We propose a Temporal Difference Learning via Frequency-Spatial Interaction (TDFSI) framework for remote sensing change detection. Specifically, we first design a Temporal Difference Enhancement Module (TDEM) to strengthen change-sensitive discrepancy cues from bi-temporal features while suppressing unstable background responses. Then, a Frequency-domain Change Refinement (FCR) module is introduced to refine the learned temporal difference representations in the spectral domain, so as to enforce global consistency, reduce pseudo-change noise, and sharpen object boundaries. Finally, the refined multi-level features are fed into a UPerNet-based change decoder to generate the final prediction.

(Insert your model architecture image here)
![Overview](path_to_your_architecture_image.png)

🚀 Environment Setup:
The TDFSI model is implemented in the PyTorch framework.

Training was conducted on two NVIDIA GTX 3090 GPUs with 24GB.

Python >= 3.7 (Recommended)

torchvision >= 0.9.0 (Recommended)

🧠 Dataset Download:
We evaluate the performance of the proposed TDFSI model on two benchmark datasets:

SYSU-CD: https://github.com/liumency/SYSU-CD

LEVIR-CD: https://justchenhao.github.io/LEVIR/

🍥 Acknowledgments:
This project utilizes the pre-trained PVTv2 as the encoder to extract initial features. Furthermore, a UPerNet-based change decoder is adopted to progressively fuse features across different levels. Thanks for their excellent works!!
