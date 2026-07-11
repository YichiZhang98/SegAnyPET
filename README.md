# SegAnyPET: Towards Universal Segmentation from 3D Whole-Body Positron Emission Tomography


This is the official repository for "Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography" [[paper]](https://arxiv.org/pdf/2603.11627) and "SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images" [[paper]](https://arxiv.org/pdf/2502.14351).

## News

*  2026.03: We conducted a thorough assessment of SegAnyPET variants on multi-center, multi-tracer, multi-disease datasets with evaluation of clinical utility in downstream applications. Please refer to [the paper](https://arxiv.org/pdf/2603.11627) for more details.
*  2026.03: We release an updated version [SegAnyPETv2](https://huggingface.co/YichiZhang98/SegAnyPET/tree/main) `(seganypet_v2.pth)` which is developed on a wider range of 11,041 multi-center whole-body PET images with a broader spectrum for organ and lesion segmentation, and a specialized variant [SegAnyPET-Lesion](https://huggingface.co/YichiZhang98/SegAnyPET/tree/main) `(seganypet_lesion.pth)` by fine-tuning SegAnyPETv2 on lesion-centric training data.
*  2025.08: We release [AutoPET-Organ]((#autopet-organ)), a dataset of 100 cases from AutoPET with expert-examined annotation of 12 organs involved in our study.
*  2025.07: We release the weights of [SegAnyPET](https://huggingface.co/YichiZhang98/SegAnyPET/tree/main) `(seganypet_v1.pth)`, the first foundation model for universal promptable 3D PET segmentation, which is trained on a collection of 5,731 whole-body PET images.
*  2025.06: SegAnyPET is accepted by [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_SegAnyPET_Universal_Promptable_Segmentation_from_Positron_Emission_Tomography_Images_ICCV_2025_paper.pdf).



## Overview

*  Positron emission tomography (PET) is an indispensable functional molecular imaging modality for oncology, neurology, and systemic disease research. Accurate volumetric segmentation of organs and lesions from PET volumes is essential for comprehensive and quantitative multi-systemic analysis of interactions between different organs and pathologies.
*  The emergence of foundation models has significantly advanced research related to medical imaging, yet their application and extension into the field of PET imaging remain largely unexplored to date. Due to the inherent modality discrepancies and distribution shifts between structural and functional imaging, existing segmentation foundational models exhibit limited transferability to PET imaging tasks.
*  To bridge this gap, we develop SegAnyPET as generalist foundational models for universal segmentation from 3D whole-body PET imaging, which generalizes robustly to unseen centers, new tracers, and novel disease types.

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/SegAnyPET.png)


## 📚 Data  <div id="autopet-organ"></div>

* We are in the progress of organizing and releasing more data to support and accelerate research in this field. Stay tuned for updates!
* We have now released the labels of AutoPET-Organ dataset at `AutoPET-OrganlabelsTr.zip`. AutoPET-Organ is a small subset of 100 cases of AutoPET with all 12 testing organs involved in our study, including liver, left kidney, right kidney, heart, spleen, aorta, prostate, left lung lower lobe, right lung lower lobe, left lung upper lobe, right lung upper lobe, and right lung middle lobe. The original images can be acquired following the [official website](https://autopet.grand-challenge.org).


## :link: Checkpoint

* We provide model checkpoints of SegAnyPET at [Hugging Face](https://huggingface.co/YichiZhang98/SegAnyPET). 


## 🔨 Usage

* SegAnyPET is adapted from [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) and retains the core functionality and usage patterns, while being optimized for universal segmentation from Positron Emission Tomography images effectively. 
* As the model is designed for promptable segmentation, ground-truth labels are required to generate prompt points for evaluation. If you want to inference an image without ground-truth, please generate pseudo mask for the target region.

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.7 (for GPU inference)
- ~4 GB GPU memory for single-case inference

### Validation

To run end-to-end validation on a test dataset:

```bash
python code/validation.py \
    --checkpoint seganypet_v2.pth \
    --test_data_path /path/to/test_data/ \
    --output_dir ./results \
    --num_clicks 5 \
    --gpu 0
```

The test data directory should be organized in nnUNet format:
```
test_data/
├── imagesTs/
│   ├── case_001.nii.gz
│   └── case_002.nii.gz
└── labelsTs/
    ├── case_001.nii.gz
    └── case_002.nii.gz
```

### Notes on Prompt-Based Inference

- SegAnyPET uses an **interactive prompt mechanism**: ground-truth labels are required to generate point prompts for evaluation (simulating user clicks).
- For inference **without ground-truth**, you can provide a pseudo-mask (e.g., from thresholding or a coarse segmentation) as the `--label` input.
- Increasing `--num_clicks` generally improves segmentation quality at the cost of longer runtime.



## 🗼 Method

#### [[arXiv'26]](https://arxiv.org/pdf/2603.11627) Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

Yichi Zhang*, Le Xue*, Wenbo Zhang, Lanlan Li, Feiyang Xiao, Yuchen Liu, Xiaohui Zhang, Hongwei Zhang, Shuqi Wang, Gang Feng, Liling Peng, Xin Gao, Yuanfan Xu, Yuan Qi, Kuangyu Shi, Hong Zhang<sup>✝</sup>, Yuan Cheng<sup>✝</sup>, Mei Tian<sup>✝</sup>, Zixin Hu<sup>✝</sup>.


#### [[ICCV'25]](https://arxiv.org/pdf/2502.14351) SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images 

Yichi Zhang*, Le Xue*, Wenbo Zhang, Lanlan Li, Yuchen Liu, Chen Jiang, Yuan Cheng<sup>✝</sup>, Yuan Qi<sup>✝</sup>.

(* Equal Contribution. <sup>✝</sup> Corresponding authors.)

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Overview.png)

![image](https://github.com/YichiZhang98/SegAnyPET/blob/main/fig/Segmentation.png)





## :books: Citation

If you find this repository helpful, please consider citing:
```
@article{zhang2026developing,
  title={Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography},
  author={Zhang, Yichi and Xue, Le and Zhang, Wenbo and Li, Lanlan and Xiao, Feiyang and Liu, Yuchen and Zhang, Xiaohui and Zhang, Hongwei and Wang, Shuqi and Feng, Gang and Peng, Liling and Gao, Xin and Xu, Yuanfan and Qi, Yuan and Shi, Kuangyu and Zhang, Hong and Cheng, Yuan and Tian, Mei and Hu, Zixin},
  journal={arXiv preprint arXiv:2603.11627},
  year={2026},
}

@inproceedings{zhang2025seganypet,
  title={SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images},
  author={Zhang, Yichi and Xue, Le and Zhang, Wenbo and Li, Lanlan and Liu, Yuchen and Jiang, Chen and Cheng, Yuan and Qi, Yuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month={October},
  year={2025},
  pages={21107-21116}
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
