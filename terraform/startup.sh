apt-get update
apt-get install -y redis-tools htop git inux-headers-`uname -r` build-essential dkms
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install ImageMagick dependencies
apt-get install -y wget autoconf pkg-config build-essential curl libpng-dev ghostscript libgs-dev libpdf-dev ocl-icd-opencl-dev

# Download and install ImageMagick
wget https://github.com/ImageMagick/ImageMagick/archive/refs/tags/7.1.0-38.tar.gz
tar xzf 7.1.0-38.tar.gz
rm 7.1.0-38.tar.gz
cd ImageMagick-7.1.0-38
./configure --prefix=/usr/local \
    --with-bzlib=yes \
    --with-fontconfig=yes \
    --with-freetype=yes \
    --with-gslib=yes \
    --with-gvc=yes \
    --with-jpeg=yes \
    --with-jp2=yes \
    --with-png=yes \
    --with-tiff=yes \
    --with-xml=yes \
    --with-gs-font-dir=yes \
    --with-gslib=yes \
    --with-pdf=yes \
    --enable-opencl
make -j
make install
ldconfig /usr/local/lib
cd ..
rm -rf ImageMagick-7.1.0-38

# Enable OpenCL at runtime
echo 'export MAGICK_OCL_DEVICE=true' >> /etc/profile.d/imagemagick.sh


# Install Docker
apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
apt-get install python3-venv
usermod -aG docker debian

# Pull chunkr git repository
git clone https://github.com/lumina-ai-inc/chunkr.git ./chunkr
chown -R debian:debian ./chunkr

