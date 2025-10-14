# SegAnyPET
The official repo of "[SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](https://arxiv.org/pdf/2502.14351)" accepted at ICCV 2025.

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Overview.png)

## Overview

* SegAnyPET is a modality-specific foundational model for universal promptable segmentation from 3D PET images. SegAnyPET is trained on a collection of 5,000+ whole-body PET images with a broader spectrum of multi-organ segmentation which can be generalized for universal segmentation from PET images.

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Segmentation.png)



## 📚 Data

We have now released the labels of AutoPET-Organ dataset at `AutoPET-OrganlabelsTr.zip`. AutoPET-Organ is a small subset of 100 cases of AutoPET with all 12 testing organs involved in our study, including liver, left kidney, right kidney, heart, spleen, aorta, prostate, left lung lower lobe, right lung lower lobe, left lung upper lobe, right lung upper lobe, and right lung middle lobe. The original images can be acquired following the [official website](https://autopet.grand-challenge.org).

## :link: Checkpoint

* We provide model checkpoints of SegAnyPET at [Hugging Face](https://huggingface.co/YichiZhang98/SegAnyPET). 


## 🔨 Usage

* SegAnyPET is adapted from [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) and retains the core functionality and usage patterns, while being optimized for universal segmentation from Positron Emission Tomography images effectively. 
* As the model is designed for promptable segmentation, ground-truth labels are required to generate prompt points for evaluation. If you want to inference an image without ground-truth, please generate pseudo mask for the target region.


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

If you use the AutoPET-Organ dataset, please also consider citing:
```
@article{gatidis2022autopet,
  title={A whole-body FDG-PET/CT dataset with manually annotated tumor lesions},
  author={Gatidis, Sergios and Hepp, Tobias and Fr{\"u}h, Marcel and La Foug{\`e}re, Christian and Nikolaou, Konstantin and Pfannenberg, Christina and Sch{\"o}lkopf, Bernhard and K{\"u}stner, Thomas and Cyran, Clemens and Rubin, Daniel},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={601},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
