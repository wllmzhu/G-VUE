# G-VUE

We present General-purpose Visual Understanding Evaluation (G-VUE), a comprehensive benchmark covering the full spectrum of visual cognitive abilities with four disjoint functional domains —Perceive, Ground, Reason, and Act. G-VUE provides a path toward a general-purpose vision system and allows for fair comparisons between different visual representations over a full spectrum of visual tasks. Specifically, Perceive tests a model's geometry understanding. Ground examines a model's acquisition of visual semantics. Reason probes a model's capacity for logical deduction and common sense reasoning. Act investigates a model's ability for planning and decision-making by learning visual policies. The four domains are embodied in 11 carefully curated tasks, from 3D reconstruction to visual reasoning and navigation. Along with the benchmark, we also provide a general encoder-decoder framework for the tasks in G-VUE. This enables any arbitrary visual representation to be used to accomplish all the 11 tasks. 


# Tasks

**Category 1: Perceive** 

* **Task 1.1:** Depth 
  * NYU v2
  * `image` → `[H,W] dense map `
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/2-1.png" width="50" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/2-2.png" width="50" height="50">
* **Task 1.2:** Camera Pose Estimation 
  * Cambridge Landmark & 7 Scene
  * `image` → `7 numbers `
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/4-1.png" width="50" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/4-2.png" width="220" height="30">
* **Task 1.3:** 3D Reconstruction
  * ShapeNet
  * `image` → `[D,H,W] dense cube `
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/1-1.png" width="50" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/1-2.png" width="50" height="50">

**Category 2: Ground** 

* **Task 2.1:** Image-Text Retrieval
  * Flickr30k
  * `image + text` → `matching score`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/9-1.png" width="200" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/9-2.png" width="200" height="30">
* **Task 2.2:** Phrase Grounding
  * RefCOCO
  * `image + text` → `4 numbers`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/5-1.png" width="200" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/5-2.png" width="220" height="30">
* **Task 2.3:** Semantic Segmentation
  * ADE20k
  * `image` → `[H,W] dense map `
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/3-1.png" width="50" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/3-2.png" width="50" height="50">

**Category 3: Reason** 

* **Task 3.1:** Visual Question Answering 
  * GQA
  * `image + text` → `class label (vocab)`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/6-1.png" width="200" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/6-2.png" width="200" height="30">
* **Task 3.2:** Common Sense Reasoning 
  * VCR
  * `image + text` → `class label`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/8-1.png" width="200" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/8-2.png" width="200" height="30">
* **Task 3.3:** Abstract Reasoning 
  * Bongard-HOI
  * `images` → `class label (binary)`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/7-1.png" width="170" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/7-2.png" width="200" height="30">

**Category 4: Act** 

* **Task 4.1:** Navigation 
  * Room to Room
  * `image` → `class label (direction)`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/10-1.png" width="200" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/10-2.png" width="200" height="50">
  * index of the next neighboring viewpoint to move to, out of all neighbors
* **Task 4.2:** Manipulation 
  * CLIPort
  * `image + text` → `pick and place (action)`
  * <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/11-1.png" width="220" height="50"> → <img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/11-2.png" width="210" height="50">
  * where and how to manipulate, as determined by dense affordance prediction

# Model Architecture

<img src="https://github.com/wllmzhu/G-VUE/blob/main/github/readme/framework_image.jpg" width="800" height="550">

# Visual Representations
* **ResNet-ImageNet**
* **ResNet-MoCo**
* **ResNet-CLIP**
* **ResNet-Ego**
* **ViT-32-CLIP**
* **ViT-16-CLIP**
* **ViT-16-MAE**

# Setup

For setup instructions, please see the `setup/` directory.

# Repo Organization

`run/`

* Directory contains bash scripts for training each task.


`base_scripts/`

* Directory contains train.py, local_eval.py and other base train/eval python scripts.


`SUBMIT.py`

* Generate submission files to be submitted to online benchmark and evaluated.


`models/`

* Directory contains the encoder-decoder model implementation.


`datasets/`

* Directory contains task dataset implementations. For Navigation, dataset/environment is located in the `models\` directory.


`configs/`

* Directory contains hydra config files used in configuring the entire repo, including models, datasets, and training.


`setup/`

* (Work-in-progress, currently refactoring and moving stuff here). Directory contains setup scripts and related code.


`transforms/`

* Utils for `datasets/`. Implementations of dataset-specific image, text, and label transforms.


`utils/`

* General utils.




