pip install gdown --user
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ../data/hicodet
cd ../data/hicodet

# Annotations: download hico.zip
gdown 'https://drive.google.com/uc?id=1BanIpXb8UH-VsA9yg4H9qlDSR0fJkBtW'
unzip hico.zip
rm hico.zip

# Move everything one level up, delete old directory
mv hico/* .
rmdir hico

# # # Images: download hico_20160224_det.tar.gz
# # gdown 'https://drive.google.com/uc?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk'
# # tar -xf hico_20160224_det.tar.gz
# # rm hico_20160224_det.tar.gz

# Extract image directory
mv hico_20160224_det.tar.gz/images .
