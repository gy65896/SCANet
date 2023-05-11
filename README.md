# [CVPRW 2023] SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2304.08444)

---
>**SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing**<br>  Yu Guo, Yuan Gao, [Ryan Wen Liu<sup>*</sup>](http://mipc.whut.edu.cn/index.html), Yuxu Lu, Jingxiang Qu, [Shengfeng He](http://www.shengfenghe.com/), [Wenqi Ren](https://sites.google.com/view/wenqiren/homepage) (* indicates corresponding author) <br> 
>IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops

> **Abstract:** *The presence of non-homogeneous haze can cause scene blurring, color distortion, low contrast, and other degradations that obscure texture details. Existing homogeneous dehazing methods struggle to handle the non-uniform distribution of haze in a robust manner. The crucial challenge of non-homogeneous dehazing is to effectively extract the non-uniform distribution features and reconstruct the details of hazy areas with high quality. In this paper, we propose a novel self-paced semi-curricular attention network, called SCANet, for non-homogeneous image dehazing that focuses on enhancing haze-occluded regions. Our approach consists of an attention generator network and a scene reconstruction network. We use the luminance differences of images to restrict the attention map and introduce a self-paced semi-curricular learning strategy to reduce learning ambiguity in the early stages of training. Extensive quantitative and qualitative experiments demonstrate that our SCANet outperforms many state-of-the-art methods. The code is publicly available at https://github.com/gy65896/SCANet.*
<hr />

## Requirement

- Python 3.7
- Pytorch 1.9.1

## Network Architecture
![fig_scanet](https://user-images.githubusercontent.com/48637474/232728784-74728cd8-c18e-40b8-a275-1b2ca24a05e7.png)

## Train

* Place the training and test image pairs in the `data` folder.
* Run `data/makedataset.py` to generate the `NH-Haze20-21-23.h5` file.
* Run `train.py` to start training.

## Test

* Place the pre-training weight in the `checkpoint` folder.
* Place test hazy images in the `input` folder.
* Modify the weight name in the `test.py`.<br> 
```
parser.add_argument("--model_name", type=str, default='Gmodel_40', help='model name')
```
* Run `test.py`
* The results is saved in `output` folder.



## Pre-training Weight Download

* The [weight40](https://drive.google.com/file/d/15-M7bGwZkXtCato_kEfLi1VOq-tjblPL/view?usp=share_link) for the NTIRE2023 val/test datasets, i.e., the weight used in the NTIRE2023 challenge.
* The [weight105](https://drive.google.com/file/d/1ATye3j81n62VHXwGihShazYnMoEbTMLd/view?usp=share_link) for the NTIRE2020/2021/2023 datasets.
* The [weight120](https://drive.google.com/file/d/1sC81YfqOa82irk_Dy37I9oxpX4zniS2z/view?usp=share_link) for the NTIRE2020/2021/2023 datasets (Add the 15 tested images as the training dataset).

<iframe frameborder="0" class="juxtapose" width="320" height="180" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=7f5bc3f6-efef-11ed-b5bd-6595d9b17862"></iframe>
## Citation

```
@article{guo2023asynchronous,
  title={SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing},
  author={Guo, Yu and Gao, Yuan and Liu, Ryan Wen and Lu, Yuxu and Qu, Jingxiang and He, Shengfeng, and Ren, Wenqi},
  journal={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
}
```

</div>
<p align="center"> 
  Visitor count<br>
  <img src="https://profile-counter.glitch.me/gy65896_SCANet/count.svg" />
</p>
