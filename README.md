# G-VUE
General-purpose Vision Understanding Evaluation


# Setup

1. Create environment from "environment.yaml".

    ```bash
    conda env create -f environment.yml
    ```

2. Install API of [CLIP](https://github.com/openai/CLIP) for initializing baseline backbones.

    ```bash
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    ```

3. Specify your pytorch version and cuda version to Install mmcv package. Take our versions for example.

    ```bash
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    ```

4. Environment for Act:navigation task

    Clone the modified Matterport3D repo and run the setup script.

    ```bash
    git clone --recursive https://github.com/wllmzhu/G-VUE-Matterport3D
    cd G-VUE-Matterport3D
    bash G-VUE-setup.sh
    ```

    As instructed by the setup script, copy the Matterport3D repo directory to the r2r config yaml's "mattersim_repo" field.

    Specify a directory to download all the required data in the r2r config yaml's "downloaded_data" field,

    then run the setup_navigation.sh script and feed in this \<data_directory\>

    ```bash
    bash scripts/setup_navigation.sh <data_directory>
    ```


5. Environment for Act:manipulation task. Please refer to [CLIPort](https://github.com/cliport/cliport) for details.

    ```bash
    git clone https://github.com/cliport/cliport.git
    cd cliport
    ```

    Note: make following preparations before installing `cliport`
      - run `export CLIPORT_ROOT=${path_to_cliport_repo}`
      - make sure to comment or delete the line `install_requires` in `setup.py`, to prevent from disturbing packages

    ```bash
    python setup.py develop
    ```

# Tasks

**Category 1: Perceive** 

* [ ] **Task 1.1:** Depth (NYU v2)
  * `image` → `[H,W] dense map `
  * depth on each pixel
* [ ] **Task 1.2:** Camera Pose Estimation (Cambridge Landmark & 7 Scene)
  * `image` → `7 numbers `
  * 3 for translation and 4 for orientation
* [ ] **Task 1.3:** 3D Reconstruction (ShapeNet)
  * `image` → `[D,H,W] dense cube `
  * Volumetric SDF

**Category 2: Ground** 

* [ ] **Task 2.1:** Image-Text Retrieval (Flickr30k)
  * `image + text` → `matching score`
  * cross-modal similarity
* [ ] **Task 2.2:** Phrase Grounding (RefCOCO)
  * `image + text` → `4 numbers`
  * bounding box representation
* [ ] **Task 2.3:** Semantic Segmentation (ADE20k)
  * `image` → `[H,W] dense map `
  * class label on each pixel

**Category 3: Reason** 

* [ ] **Task 3.1:** Visual Question Answering (GQA)
  * `image + text` → `class label (vocab)`
  * one or two word short response to question
* [ ] **Task 3.2:** Common Sense Reasoning (VCR)
  * `image + text` → `class label`
  * selection among 4 answer candidates
* [ ] **Task 3.3:** Abstract Reasoning (Bongard-HOI)
  * `images` → `class label (binary)`
  * positive class or negative class regarding shot samples

**Category 4: Act** 

* [ ] **Task 4.1:** Navigation (R2R)
  * `image` → `class label (direction)`
  * index, out of all neighbors, of the next one to move to
* [ ] **Task 4.2:** Manipulation (CLIPort)
  * `image + text` → `pick and place (action)`
  * where and how to manipulation are determined by dense affordance prediction
