DATA_DIR=$1
DOWNLOAD_SCRIPT=preprocess/r2r/download-mp.py

# If the required Matterport3D dataset download script is not found
if ! test -f "$DOWNLOAD_SCRIPT" 
then
    echo "Please go to https://niessner.github.io/Matterport/ \n
            Follow instruction there to email the Matterport3D team your signed Terms of Use \n
            After obtaining a reply from them, copy the attached dowload-mp.py script to \n
            G-VUE/preprocess/r2r \n"
fi

# Create specified data directory if doesn't exist
mkdir -p $DATA_DIR

# Run the download script 
python2 $DOWNLOAD_SCRIPT -o $DATA_DIR --type undistorted_camera_parameters matterport_skybox_images 

# Unzip the downloaded data
CURR_DIR=$(pwd)
cd ${DATA_DIR}/v1/scans
bash ${CURR_DIR}/preprocess/r2r/unzip_matterport.sh
cd CURR_DIR

# Resize skyboxes to (224,224) to decrease loading time. Script modified from the original Matterport repo
python preprocess/r2r/downsize_skybox.py


