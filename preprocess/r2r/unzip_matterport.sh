#go to the v1/scan directory, then run this simple script to extract all images

for d in */ ; do
    cd $d
    unzip *
    mv $d* .
    rmdir $d
    cd ..
done