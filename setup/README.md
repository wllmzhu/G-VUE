# Setup

## Environment for Act:Navigation task

1. Clone the [modified Matterport3D repo](https://github.com/wllmzhu/G-VUE-Matterport3D) and run its setup script.

   ```bash
   git clone --recursive https://github.com/wllmzhu/G-VUE-Matterport3D
   cd G-VUE-Matterport3D
   bash G-VUE-setup.sh
   ```

2. As instructed by the setup script, copy the `Matterport3D` repo directory to the fill in the `mattersim_repo` field in `configs/r2r.yaml`.

3. Download the data to `<data_directory>`, which is specified in the `downloaded_data` field in `configs/r2r.yaml`.

4. Run the `setup_navigation.sh` script with the same `<data_directory>`.

   ```bash
   bash setup/setup_navigation.sh <data_directory>
   ```


## Environment for Act:Manipulation task

The setup procedures are similar to [CLIPort](https://github.com/cliport/cliport), which you can refer to for more details. The simplified procedures are as follows:

1. Clone repo.

   ```bash
   git clone https://github.com/cliport/cliport.git
   ```

2. Specify the environment variable of path before installing `CLIPort`.

   ```bash
   cd cliport
   export CLIPORT_ROOT=${pwd}
   ```

3. Install. If installation errors are raised by the packages in `install_requires` in `setup.py`, try to comment or delete the corresponding line.

    ```bash
    python setup.py develop
    ```
