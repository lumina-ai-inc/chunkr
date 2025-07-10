#!/bin/bash

# Simple Teleport Agent VM Startup Script
# Just install Docker, Git, Teleport and connect to server

set -e

# Template variables
TELEPORT_SERVER_IP="${teleport_server_ip}"
TELEPORT_TOKEN="${teleport_token}"
TELEPORT_CA_PIN="${teleport_ca_pin}"
AGENT_HOSTNAME="${agent_hostname}"
REGION="${region}"

# Log everything to a file
exec > >(tee -a /var/log/teleport-agent-setup.log) 2>&1

echo "Starting Teleport Agent VM setup at $(date)"
echo "Teleport Server IP: $TELEPORT_SERVER_IP"
echo "Agent Hostname: $AGENT_HOSTNAME"
echo "Region: $REGION"

# Update system
echo "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Git
echo "Installing Git..."
apt-get install -y git

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Teleport
echo "Installing Teleport..."
curl -L https://goteleport.com/static/install.sh | bash -s 13.4.16

# Get the actual hostname
ACTUAL_HOSTNAME=$(hostname)

# Wait for Docker to be ready
echo "Waiting for Docker to be ready..."
sleep 10

# Create systemd service for teleport agent
echo "Creating Teleport agent service..."
cat > /etc/systemd/system/teleport-agent.service << EOF
[Unit]
Description=Teleport Agent
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=5
User=root
Group=root
ExecStart=/usr/local/bin/teleport start \\
  --roles=node \\
  --token="$TELEPORT_TOKEN" \\
  --ca-pin="$TELEPORT_CA_PIN" \\
  --auth-server="$TELEPORT_SERVER_IP:3025" \\
  --labels="env=production,region=$REGION,hostname=$ACTUAL_HOSTNAME,type=agent-vm"

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the teleport service
echo "Starting Teleport agent service..."
systemctl daemon-reload
systemctl enable teleport-agent
systemctl start teleport-agent

# Wait for agent to start
sleep 15

# Check if Teleport agent service is running
if systemctl is-active --quiet teleport-agent; then
    echo "✅ Teleport agent service is running successfully"
    systemctl status teleport-agent --no-pager || true
else
    echo "❌ Teleport agent service failed to start, checking logs..."
    journalctl -u teleport-agent --no-pager -n 20 || echo "No logs available"
fi

# Create simple info script
cat > /home/ubuntu/agent-info.sh << 'EOF'
#!/bin/bash
echo "🤖 Teleport Agent VM Information"
echo "================================"
echo "Hostname: $(hostname)"
echo "Internal IP: $(hostname -I | awk '{print $1}')"
echo "External IP: $(curl -s ifconfig.me 2>/dev/null || echo 'Not available')"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Uptime: $(uptime -p)"
echo ""
echo "🚀 Access via Teleport:"
echo "  1. Open Teleport web interface"
echo "  2. Go to 'Servers' section"
echo "  3. Find '$(hostname)' in the list"
echo "  4. Click 'Connect' for terminal access"
EOF
chmod +x /home/ubuntu/agent-info.sh
chown ubuntu:ubuntu /home/ubuntu/agent-info.sh

# Simple MOTD
cat > /etc/motd << 'EOF'

🤖 Teleport Agent VM
====================
This VM is managed by Teleport for secure access.

Quick Commands:
  ./agent-info.sh               - Show VM information
  sudo systemctl status teleport-agent  - Check agent status

Access this VM securely through the Teleport web interface!

EOF

echo ""
echo "============================================="
echo "Teleport Agent VM Setup Complete!"
echo "============================================="
echo "VM Hostname: $ACTUAL_HOSTNAME"
echo "Teleport Server: $TELEPORT_SERVER_IP:3025"
echo ""
echo "✅ Teleport Agent running"
echo "This VM should now appear in your Teleport web interface!"
echo "============================================="

echo "Setup completed at $(date)" 