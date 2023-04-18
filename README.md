# [CVPRW 2023] SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing

---
>**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing [paper](http://arxiv.org/abs/2304.08444)**<br>  Yu Guo, Yuan Gao, Ryan Wen Liu<sup>*</sup>, Yuxu Lu, Jingxiang Qu, [Shengfeng He](http://www.shengfenghe.com/), [Wenqi Ren](https://it.ouc.edu.cn/djy_23898/main.htm) (* indicates corresponding author)<br>
>IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops

## Preparation

- Python 3.7
- Pytorch 1.9.1

## Abstract
The presence of non-homogeneous haze can cause scene blurring, color distortion, low contrast, and other degradations that obscure texture details. Existing homogeneous dehazing methods struggle to handle the non-uniform distribution of haze in a robust manner. The crucial challenge of non-homogeneous dehazing is to effectively extract the non-uniform distribution features and reconstruct the details of hazy areas with high quality. In this paper, we propose a novel self-paced semi-curricular attention network, called SCANet, for non-homogeneous image dehazing that focuses on enhancing haze-occluded regions. Our approach consists of an attention generator network and a scene reconstruction network. We use the luminance differences of images to restrict the attention map and introduce a self-paced semi-curricular learning strategy to reduce learning ambiguity in the early stages of training. Extensive quantitative and qualitative experiments demonstrate that our SCANet outperforms many state-of-the-art methods. The code is publicly available at https://github.com/gy65896/SCANet.


## Network Architecture
