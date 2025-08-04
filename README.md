
<p align="center">
  <h2 align="center"> <img src='assets/logo.jpg' width='24px' />Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation</h2>
  <p align="center">
    <a href=https://github.com/xinchengshuai/><strong>Xincheng Shuai</strong></a><sup>1</sup>
    ·
    <a href=https://henghuiding.com/><strong>Henghui Ding </strong></a><sup>1</sup>
        .
    <a href=https://github.com/qqzy/><strong>Zhenyuan Qin</strong></a><sup>1</sup>
    .
    <a href=https://scholar.google.com/citations?user=7QvWnzMAAAAJ&hl/><strong>Hao Luo</strong></a><sup>2,3</sup>
    ·
    <a href=https://scholar.google.com/citations?user=XQViiyYAAAAJ&hl/><strong>Xingjun Ma</strong></a><sup>1</sup>
    .
    <a href=https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl><strong>Dacheng Tao</strong></a><sup>4</sup>

</p>

<p align="center">
    <sup>1</sup>Fudan University · <sup>2</sup>DAMO Academy, Alibaba group  · <sup>3</sup>Hupan Lab  · <sup>4</sup>Nanyang Technological University, Singapore
</p>
<p align="center">
<a href="https://arxiv.org/abs/2501.01425"><img src="https://img.shields.io/static/v1?label=Paper&message=2505.19114&color=red&logo=arxiv"></a>
<a href="https://henghuiding.com/SynFMC/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>
<a href="https://huggingface.co/datasets/XinchengShuai/SynFMC"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace">
<!-- <img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"> -->
</p>





## 🎯 Introduction
Controlling the movements of dynamic objects and the camera within generated videos is a meaningful yet challenging task. Due to the lack of datasets with comprehensive 6D pose annotations, existing text-to-video methods can not simultaneously control the motions of both camera and objects in 3D-aware manner. Therefore we introduce a Synthetic Dataset for Free-Form Motion Control (SynFMC). The proposed SynFMC dataset includes diverse object and environment categories and covers various motion patterns according to specific rules, simulating common and complex real-world scenarios. The complete 6D pose information facilitates models learning to disentangle the motion effects from objects and the camera in a video. To provide precise 3D-aware motion control, we further propose a method trained on SynFMC, Free-Form Motion Control (FMC). FMC can control the 6D poses of objects and camera independently or simultaneously, producing high-fidelity videos.

<img src='assets/teaser.svg' width='100%' />
<p align="left">
   <b>Figure 1.</b> The rule-based generation pipeline of videos in the proposed Synthetic Dataset for Free-Form Motion Control (SynFMC). This example generates synthetic video with three objects: (1) The environment asset and it’s matching object assets are selected as the scene elements. (2) The motion types of objects and camera are randomly selected for trajectory generation. (3) The center region shows the resulting 3D animation sequence used for rendering. The rendered video and annotations are demonstrated in the last row.
</p>
<br>

<img src='assets/network.svg' width='100%' />
<p align="left">
   <b>Figure 2.</b> The architecture of FMC. In the first stage, we randomly sample the images from synthetic videos and update the parameters from injected Domain LoRA. Next, the modules from CMC are learned. It consists of two parts: Camera Encoder and Camera Adapter, where the Camera Adapter is introduced into the temporal modules. Finally, we train the Object Encoder from OMC. It receives the 6D object pose features, which are repeated in the corresponding object region. We use Gaussian blur kernel centered at the centroid to prevent the need of precise masks. Then, the output is multiplied by the coarse masks to modulate the features in the main branch.
</p>
<br>



## ⚙️Quick Start
### 1. Setup
```bash
conda env create -f environment.yaml
conda activate fmc
```

### 2. Training
The training process of FMC consists of three stages.
#### 2.1 Learn Domain LoRA
In the first stage, we randomly sample the images from synthetic videos and update the parameters from injected Domain LoRA. 
```bash
bash dist_run_lora.bash
```

#### 2.2 Learn Camera Motion Controller (CMC)
Next, the modules from CMC are learned. Inspired by Cameractrl, it consists of two parts: Camera Encoder and Camera Adapter, where the Camera Adapter is introduced into the temporal modules. 
```bash
bash dist_run_cam.bash
```

#### 2.3 Learn Object Motion Controller (OMC)
Finally, we train the Object Encoder from OMC. It receives the 6D object pose features, which are repeated in the corresponding object region. We use Gaussian blur kernel centered at the centroid to prevent the need of precise masks. Then, the output is multiplied by the coarse masks to modulate the features in the main branch.
```bash
bash dist_run_obj.bash
```

<br>

<!-- ## 📋 TODO List
- [x] Upload training code of FMC.
- [ ] Upload SynFMC dataset (in progress).
- [ ] Upload the code of SynFMC.
- [ ] Upload inference code and model weights of FMC. -->

## ✒️ Citation

If you find our work useful for your research and applications, please kindly cite using this BibTeX:

```latex
@inproceedings{SynFMC,
        title={{Free-Form Motion Control}: Controlling the 6D Poses of Camera and Objects in Video Generation},
        author={Shuai, Xincheng and Ding, Henghui and Qin, Zhenyuan and Luo, Hao and Ma, Xingjun and Tao, Dacheng},
        booktitle={ICCV},
        year={2025}
      }
```
