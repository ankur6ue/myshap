# This script builds the rayproject/ray and other images. This image can then be used as a base to
# build your application image. See Dockerfile in root folder
#git clone --branch releases/1.0.0 https://github.com/ray-project/ray.git
rm -rf ray
git clone https://github.com/ray-project/ray.git
cd ray
./build-docker.sh