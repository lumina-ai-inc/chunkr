apt-get update
apt-get install -y redis-tools htop git inux-headers-`uname -r` build-essential dkms
wget https://us.download.nvidia.com/tesla/535.161.07/NVIDIA-Linux-x86_64-535.161.07.run
chmod +x NVIDIA-Linux-x86_64-535.161.07.run
sh NVIDIA-Linux-x86_64-535.161.07.run

# Install Docker
apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
apt-get install python3-venv
usermod -aG docker debian

# Pull chunkr git repository
git clone https://github.com/lumina-ai-inc/chunkr.git /home/debian/chunkr
chown -R debian:debian /home/debian/chunkr