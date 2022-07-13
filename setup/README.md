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

4. Environment for Act:Navigation task

    Clone the modified Matterport3D repo and run the setup script.

    ```bash
    git clone --recursive https://github.com/wllmzhu/G-VUE-Matterport3D
    cd G-VUE-Matterport3D
    bash G-VUE-setup.sh
    ```

    As instructed by the setup script, copy the Matterport3D repo directory to the r2r config yaml's "mattersim_repo" field.

    Specify a \<data_directory\> to download all the required data in the r2r config yaml's "downloaded_data" field,

    then run the setup_navigation.sh script and feed in this \<data_directory\>

    ```bash
    bash scripts/setup_navigation.sh <data_directory>
    ```


5. Environment for Act:Manipulation task. Please refer to [CLIPort](https://github.com/cliport/cliport) for details.

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

