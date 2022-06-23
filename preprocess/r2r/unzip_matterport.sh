# A component of setup_navigation.sh, unzip all matterport data
# Meant to be run inside v1/scan directory

for d in */ ; do
    cd $d
    unzip "*.zip"
    mv $d* .
    rmdir $d
    cd ..
done