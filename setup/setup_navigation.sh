DATA_DIR=$1
CURR_DIR=$(pwd)
DOWNLOAD_SCRIPT=setup/preprocess/r2r/download_mp.py

# Check if DATA_DIR is supplied
if [ -z "$1" ]
then
echo "No DATA_DIR is supplied"
exit 1
fi

#If the required Matterport3D dataset download script is not found
if ! test -f "$DOWNLOAD_SCRIPT" 
then
echo "Unable to find download-mp.py. Please go to https://niessner.github.io/Matterport/ ,
Follow instruction there to email the Matterport3D team your signed Terms of Use.
After they reply with the required dowload-mp.py script, copy the script to G-VUE/setup/preprocess/r2r"
exit 1
fi

# Create specified data directory if doesn't exist
mkdir -p $DATA_DIR

# Run the download script 
python2 $DOWNLOAD_SCRIPT -o $DATA_DIR --type undistorted_camera_parameters matterport_skybox_images 

# Unzip the downloaded data
cd ${DATA_DIR}/v1/scans
bash ${CURR_DIR}/preprocess/r2r/unzip_matterport.sh
cd ${CURR_DIR}

# Resize skyboxes to (224,224) to decrease loading time. Script modified from Matterport.
python preprocess/r2r/downsize_skybox.py

# Download pretrained Prevalent weights. URL obtained from https://github.com/YicongHong/Recurrent-VLN-BERT
mkdir -p ${DATA_DIR}/Prevalent/pretrained_model
cd ${DATA_DIR}/Prevalent/pretrained_model
gdown --folder https://drive.google.com/drive/folders/1sW2xVaSaciZiQ7ViKzm_KbrLD_XvOq5y
cd checkpoint-12864
mv * ..
cd ..
rmdir checkpoint-12864
cd ${CURR_DIR}

# Download R2R data. URL obtained from Matterport3D and Recurrent-VLN-BERT
mkdir -p ${DATA_DIR}/r2r_data
cd ${DATA_DIR}/r2r_data
wget https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json
wget https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json
wget https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json
wget https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json
wget https://raw.githubusercontent.com/YicongHong/Recurrent-VLN-BERT/main/data/id_paths.json
wget https://raw.githubusercontent.com/YicongHong/Recurrent-VLN-BERT/main/data/R2R_val_train_seen.json
cd ${CURR_DIR}

# Create directory to place precomputed visual features
mkdir -p ${DATA_DIR}/r2r_img_features