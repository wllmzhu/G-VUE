# G-VUE
General *Vision* Understanding Evaluation


# Setup
- Register: run command below in the top-level directory ("G-VUE") to register packages in this repo.

  ```bash
  pip install -e .
  ```

- Create environment from "environment.yaml".

  ```bash
  conda env create -f environment.yml
  ```

- Install API of [CLIP](https://github.com/openai/CLIP) for initializing baseline backbones.

  ```bash
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  ```

- Specify your pytorch version and cuda version to Install mmcv package. Take our versions for example.

  ```bash
  pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
  ```

# Tasks

**Category 1: Perceive** 

* [ ] **Task 1.1:** Depth (NYU v2)
  * `image` → `[H,W] dense map `
  * depth on each pixel
* [ ] **Task 1.2:** 6D Pose Estimation (LineMod)
  * `image` → `6 numbers `
  * 3 positions, 3 orientation
* [ ] **Task 1.3:** 3D Reconstruction (ShapeNet)
  * `image` → `[H,W,L] dense map `
  * 3D voxels

**Category 2: Ground** 

* [ ] **Task 2.1:** Segmentation (ADE20K)
  * `image` → `[H,W,C] dense map `
  * class identity on each pixel
* [ ] **Task 2.2:** Phrase Grounding (Ref-COCO)
  * `image + text` → `4 numbers`
  * bounding box coordinates
* [ ] **Task 2.3:** Vision-Language Retrieval (Flickr30k)
  * `image + text` → `class label (multiple choice)`
  * index of the right reference sentence from a choice of 5

**Category 3: Reason** 

* [ ] **Task 3.1:** Visual Question Answering (GQA)
  * `image + text` → `class label (vocab)`
  * one or two word short response to question
* [ ] **Task 3.2:** Bongard Problem (Bongard-HOI)
  * `image` → `class label (boolean)`
  * 
* [ ] **Task 3.3:** Common Sense Reasoning (VCR)
  * `image + text` → `class label`
  * 

**Category 4: Act** 

* [ ] **Task 4.1:** Affordance Reasoning (FunkPoint)
  * `image` → ` `
  * 
* [ ] **Task 4.2:** Navigation (PointGoal)
  * `image` → ` `
  * 
* [ ] **Task 4.3:** Manipulation (CLIPort)
  * `image` → ` `
  * 

