apt-get update
apt-get install -y redis-tools htop git build-essential dkms

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
