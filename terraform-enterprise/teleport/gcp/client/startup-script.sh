#!/bin/bash

# Teleport Server VM Startup Script
# This script sets up Docker and Teleport on a new VM

set -e

# Log everything to a file
exec > >(tee -a /var/log/teleport-setup.log) 2>&1

echo "Starting Teleport VM setup at $(date)"

# Update system
echo "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    openssl \
    unzip

# Install Docker
echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Docker Compose standalone
echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create teleport directory
echo "Setting up Teleport directory..."
mkdir -p /home/ubuntu/teleport
chown -R ubuntu:ubuntu /home/ubuntu/teleport

# Create the Teleport configuration directory
mkdir -p /home/ubuntu/teleport/teleport-data
mkdir -p /home/ubuntu/teleport/config
chown -R ubuntu:ubuntu /home/ubuntu/teleport

# Create docker-compose.yml for Teleport
cat > /home/ubuntu/teleport/docker-compose.yml << 'EOF'
version: '3.8'

services:
  teleport:
    image: public.ecr.aws/gravitational/teleport:13
    container_name: teleport-server
    hostname: teleport
    ports:
      - "3080:3080"   # Web interface
      - "3025:3025"   # Auth server (for agents)
      - "3022:3022"   # SSH (optional)
    volumes:
      - ./teleport-data:/var/lib/teleport
      - ./config:/etc/teleport
    environment:
      - TELEPORT_HOSTNAME=${teleport_hostname}
    restart: unless-stopped
EOF

# Create proper Teleport configuration file
cat > /home/ubuntu/teleport/config/teleport.yaml << 'EOF'
version: v3
teleport:
  nodename: teleport-server
  data_dir: /var/lib/teleport
  log:
    output: stderr
    severity: INFO
  ca_pin: ""
  diag_addr: ""
auth_service:
  enabled: "yes"
  listen_addr: 0.0.0.0:3025
  cluster_name: teleport-cluster
  public_addr: ${teleport_hostname}:3025
proxy_service:
  enabled: "yes"
  listen_addr: 0.0.0.0:3023
  web_listen_addr: 0.0.0.0:3080
  public_addr: ${teleport_hostname}:3080
  https_keypairs: []
  https_cert_file: ""
  https_key_file: ""
ssh_service:
  enabled: "yes"
  listen_addr: 0.0.0.0:3022
  public_addr: ${teleport_hostname}:3022
EOF

# Create the Teleport systemd service
cat > /etc/systemd/system/teleport-server.service << 'EOF'
[Unit]
Description=Teleport Server
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/teleport
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the Teleport service
systemctl daemon-reload
systemctl enable teleport-server.service

# Generate a secure token for agents
echo "Generating secure token..."
TELEPORT_TOKEN=$(openssl rand -hex 32)
echo "Generated Teleport token: $TELEPORT_TOKEN" >> /home/ubuntu/teleport-token.txt
chown ubuntu:ubuntu /home/ubuntu/teleport-token.txt

# Create a helper script for creating admin users
cat > /home/ubuntu/create-admin-user.sh << 'EOF'
#!/bin/bash
# Helper script to create admin user

if [ $# -ne 1 ]; then
    echo "Usage: $0 <username>"
    exit 1
fi

USERNAME=$1

echo "Creating admin user: $USERNAME"
cd /home/ubuntu/teleport

# Create admin user
docker-compose exec teleport tctl users add $USERNAME --roles=editor,access --logins=root,ubuntu,admin

echo "Admin user created successfully!"
echo "Use the signup link provided above to set up the user."
EOF

chmod +x /home/ubuntu/create-admin-user.sh
chown ubuntu:ubuntu /home/ubuntu/create-admin-user.sh

# Create a helper script for adding tokens
cat > /home/ubuntu/create-token.sh << 'EOF'
#!/bin/bash
# Helper script to create tokens for agents

cd /home/ubuntu/teleport

echo "Creating new token for agents..."
docker-compose exec teleport tctl tokens add --type=node --ttl=1h

echo ""
echo "To connect an agent, run this command on the target VM:"
echo "docker run -d --name teleport-agent --hostname \$(hostname) --net=host --pid=host -v /etc/machine-id:/etc/machine-id:ro -v /var/lib/teleport:/var/lib/teleport public.ecr.aws/gravitational/teleport:13 teleport start --roles=node --token=TOKEN_FROM_ABOVE --auth-server=${teleport_hostname}:3025 --labels=env=production"
EOF

chmod +x /home/ubuntu/create-token.sh
chown ubuntu:ubuntu /home/ubuntu/create-token.sh

# Create a helper script for managing the service
cat > /home/ubuntu/teleport-manage.sh << 'EOF'
#!/bin/bash
# Helper script to manage Teleport service

case $1 in
    start)
        echo "Starting Teleport server..."
        cd /home/ubuntu/teleport && docker-compose up -d
        ;;
    stop)
        echo "Stopping Teleport server..."
        cd /home/ubuntu/teleport && docker-compose down
        ;;
    restart)
        echo "Restarting Teleport server..."
        cd /home/ubuntu/teleport && docker-compose down && docker-compose up -d
        ;;
    logs)
        echo "Showing Teleport logs..."
        cd /home/ubuntu/teleport && docker-compose logs -f
        ;;
    status)
        echo "Teleport server status:"
        cd /home/ubuntu/teleport && docker-compose ps
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
EOF

chmod +x /home/ubuntu/teleport-manage.sh
chown ubuntu:ubuntu /home/ubuntu/teleport-manage.sh

# Create README for the user
cat > /home/ubuntu/README.md << 'EOF'
# Teleport Server Setup

This VM has been configured with Teleport server running in Docker.

## Quick Start

1. **Access the web interface:**
   - URL: https://YOUR_VM_IP:3080
   - Use the signup link from the admin user creation

2. **Create your first admin user:**
   ```bash
   ./create-admin-user.sh your-username
   ```

3. **Create tokens for agents:**
   ```bash
   ./create-token.sh
   ```

4. **Manage the service:**
   ```bash
   ./teleport-manage.sh {start|stop|restart|logs|status}
   ```

## Files and Directories

- `/home/ubuntu/teleport/` - Teleport configuration and data
- `/home/ubuntu/teleport-token.txt` - Initial secure token
- `/home/ubuntu/create-admin-user.sh` - Script to create admin users
- `/home/ubuntu/create-token.sh` - Script to create agent tokens
- `/home/ubuntu/teleport-manage.sh` - Service management script

## Connecting Agents

To connect a VM to this Teleport server:

1. Install Docker on the target VM
2. Get a token using `./create-token.sh`
3. Run the agent command provided by the token script

## Security Notes

- The web interface is accessible on port 3080
- Agents connect on port 3025
- All connections use TLS encryption
- Session recording is enabled by default

## Troubleshooting

- Check logs: `./teleport-manage.sh logs`
- Check status: `./teleport-manage.sh status`
- Check firewall: `sudo ufw status`
- Check if running: `docker ps`

## Configuration

The Teleport configuration is in `/home/ubuntu/teleport/config/teleport.yaml`.
Data is stored in `/home/ubuntu/teleport/teleport-data/`.
EOF

chown ubuntu:ubuntu /home/ubuntu/README.md

# Start Teleport service
echo "Starting Teleport service..."
systemctl start teleport-server.service

# Wait for service to start
sleep 30

# Check if Teleport is running
echo "Checking Teleport status..."
systemctl status teleport-server.service

# Show final information
echo ""
echo "============================================="
echo "Teleport VM Setup Complete!"
echo "============================================="
echo "VM IP: ${teleport_hostname}"
echo "Web Interface: https://${teleport_hostname}:3080"
echo "Agent Auth Server: ${teleport_hostname}:3025"
echo ""
echo "Next steps:"
echo "1. SSH into the VM: gcloud compute ssh [VM_NAME] --zone [ZONE]"
echo "2. Create admin user: ./create-admin-user.sh your-username"
echo "3. Access web interface and complete setup"
echo "4. Create tokens for agents: ./create-token.sh"
echo ""
echo "See /home/ubuntu/README.md for detailed instructions"
echo "============================================="

echo "Setup completed at $(date)" 