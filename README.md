# SegAnyPET
The official repo of "[SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](https://arxiv.org/pdf/2502.14351)" accepted at ICCV 2025.

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Overview.png)

## Overview

* SegAnyPET is a modality-specific foundational model for universal promptable segmentation from 3D PET images. SegAnyPET is trained on a collection of 5,000+ whole-body PET images with a broader spectrum of multi- organ segmentation which can be generalized for universal segmentation.

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Segmentation.png)


## :link: Checkpoint

* We provide model checkpoints of SegAnyPET at [Hugging Face](https://huggingface.co/YichiZhang98/SegAnyPET).


## 🔨 Usage

* Our code is adapted from [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). We are organizing our training and evaluation code and will make it publicly available soon.



## :books: Citation

If you find this repository helpful, please consider citing:
```
@article{zhang2025seganypet,
  title={SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images},
  author={Zhang, Yichi and Xue, Le and Zhang, Wenbo and Li, Lanlan and Liu, Yuchen and Jiang, Chen and Cheng, Yuan and Qi, Yuan},
  journal={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
