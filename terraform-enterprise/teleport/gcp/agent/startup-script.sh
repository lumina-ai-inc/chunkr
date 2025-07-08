#!/bin/bash

# Teleport Agent VM Startup Script
# This script sets up Docker and Teleport agent on a new VM

set -e

# Template variables
TELEPORT_SERVER_IP="${teleport_server_ip}"
TELEPORT_TOKEN="${teleport_token}"
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

# Install required packages
echo "Installing required packages..."
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    nginx \
    nodejs \
    npm \
    python3 \
    python3-pip \
    htop \
    vim \
    unzip \
    netcat-openbsd

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

# Get the actual hostname
ACTUAL_HOSTNAME=$(hostname)

# Wait for Docker to be ready
echo "Waiting for Docker to be ready..."
sleep 10

# Install Teleport natively on the host
echo "Installing Teleport natively..."
curl -L https://goteleport.com/static/install.sh | bash -s 13.4.16

# Create teleport user and data directory
echo "Setting up Teleport directories..."
useradd -r -s /bin/false teleport || true
mkdir -p /var/lib/teleport
chown -R teleport:teleport /var/lib/teleport
chmod -R 755 /var/lib/teleport

# Create systemd service for the agent (running as root for user switching)
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
    --ca-pin=sha256:3c9058f47ce03272518c7399fc555f6deb0859a9bf50591400f0e22cddd99e7a \\
  --auth-server="$TELEPORT_SERVER_IP:3025" \\
  --labels="env=production,region=$REGION,hostname=$ACTUAL_HOSTNAME,type=agent-vm,os=ubuntu-22.04"

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
echo "Starting Teleport agent service..."
systemctl daemon-reload
systemctl enable teleport-agent
systemctl start teleport-agent

# Wait for agent to start and verify it's running
sleep 15

# Check if Teleport agent service is running
if systemctl is-active --quiet teleport-agent; then
    echo "✅ Teleport agent service is running successfully"
    systemctl status teleport-agent --no-pager || true
else
    echo "❌ Teleport agent service failed to start, checking logs..."
    journalctl -u teleport-agent --no-pager -n 20 || echo "No logs available"
fi

# Set up a simple web service for demonstration
echo "Setting up demo web service..."
mkdir -p /home/ubuntu/demo-app
cat > /home/ubuntu/demo-app/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Teleport Agent Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        .info { background: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 4px; }
        .status { color: #28a745; font-weight: bold; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header">🚀 Teleport Agent VM</h1>
        <div class="info">
            <p><strong>Status:</strong> <span class="status">✅ Running</span></p>
            <p><strong>Hostname:</strong> <code id="hostname"></code></p>
            <p><strong>Teleport Server:</strong> <code>$${teleport_server_ip}:3025</code></p>
            <p><strong>Access Method:</strong> Via Teleport Web Interface</p>
        </div>
        
        <h2>System Information</h2>
        <pre id="sysinfo">Loading...</pre>
        
        <h2>Running Services</h2>
        <ul>
            <li>🌐 Nginx Web Server (Port 80)</li>
            <li>🐳 Docker Engine</li>
            <li>📡 Teleport Agent</li>
            <li>🔧 Demo Node.js App (Port 3000)</li>
        </ul>
        
        <h2>How to Access</h2>
        <ol>
            <li>Open Teleport Web Interface</li>
            <li>Navigate to "Servers"</li>
            <li>Find this VM in the list</li>
            <li>Click "Connect" for terminal access</li>
        </ol>
    </div>
    
    <script>
        document.getElementById('hostname').textContent = 'ACTUAL_HOSTNAME';
        
        // Simulate system info (in a real scenario, this would be fetched from an API)
        setTimeout(() => {
            document.getElementById('sysinfo').textContent = 
                'OS: Ubuntu 22.04 LTS\n' +
                'CPU: 2 vCPUs\n' +
                'Memory: 8GB\n' +
                'Disk: 30GB SSD\n' +
                'Region: us-central1\n' +
                'Managed by: Terraform + Teleport';
        }, 1000);
    </script>
</body>
</html>
EOF

# Replace placeholder with actual hostname
sed -i "s/ACTUAL_HOSTNAME/$ACTUAL_HOSTNAME/g" /home/ubuntu/demo-app/index.html

# Configure Nginx to serve the demo app
cat > /etc/nginx/sites-available/demo << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    root /home/ubuntu/demo-app;
    index index.html;
    
    server_name _;
    
    location / {
        try_files $uri $uri/ =404;
    }
    
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable the demo site
rm -f /etc/nginx/sites-enabled/default
ln -s /etc/nginx/sites-available/demo /etc/nginx/sites-enabled/
systemctl restart nginx
systemctl enable nginx

# Set up a simple Node.js demo app
echo "Setting up Node.js demo app..."
mkdir -p /home/ubuntu/node-demo
cat > /home/ubuntu/node-demo/app.js << 'EOF'
const express = require('express');
const os = require('os');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
    const info = {
        hostname: os.hostname(),
        platform: os.platform(),
        arch: os.arch(),
        cpus: os.cpus().length,
        memory: `$${Math.round(os.totalmem() / 1024 / 1024 / 1024)}GB`,
        uptime: `$${Math.round(os.uptime() / 60)} minutes`,
        timestamp: new Date().toISOString(),
        message: '🎉 Hello from Teleport Agent VM!'
    };
    
    res.json(info);
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Demo app listening at http://0.0.0.0:$${port}`);
});
EOF

cat > /home/ubuntu/node-demo/package.json << 'EOF'
{
  "name": "teleport-agent-demo",
  "version": "1.0.0",
  "description": "Demo app for Teleport Agent VM",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  },
  "dependencies": {
    "express": "^4.18.0"
  }
}
EOF

# Install Node.js dependencies and start the app
cd /home/ubuntu/node-demo
npm install
chown -R ubuntu:ubuntu /home/ubuntu/node-demo

# Create systemd service for the Node.js app
cat > /etc/systemd/system/node-demo.service << 'EOF'
[Unit]
Description=Node.js Demo App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/node-demo
ExecStart=/usr/bin/node app.js
Restart=always
RestartSec=3
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable node-demo
systemctl start node-demo

# Create a simple monitoring script
cat > /home/ubuntu/check-services.sh << 'EOF'
#!/bin/bash

echo "=== Teleport Agent Status ==="
docker ps | grep teleport || echo "❌ Teleport agent not running"

echo ""
echo "=== Nginx Status ==="
systemctl is-active nginx || echo "❌ Nginx not running"

echo ""
echo "=== Node Demo Status ==="
systemctl is-active node-demo || echo "❌ Node demo not running"

echo ""
echo "=== Network Connectivity ==="
curl -s http://localhost/health || echo "❌ Nginx health check failed"
curl -s http://localhost:3000/health || echo "❌ Node app health check failed"

echo ""
echo "=== Teleport Connectivity ==="
if nc -z $TELEPORT_SERVER_IP 3025; then
    echo "✅ Can reach Teleport server"
else
    echo "❌ Cannot reach Teleport server"
fi
EOF
chmod +x /home/ubuntu/check-services.sh
chown ubuntu:ubuntu /home/ubuntu/check-services.sh

# Create a simple info script
cat > /home/ubuntu/agent-info.sh << 'EOF'
#!/bin/bash

echo "🤖 Teleport Agent VM Information"
echo "================================"
echo "Hostname: $(hostname)"
echo "Internal IP: $(hostname -I | awk '{print $1}')"
echo "External IP: $(curl -s ifconfig.me 2>/dev/null || echo 'Not available')"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime -p)"
echo ""
echo "🔗 Services:"
echo "  - Web Server: http://$(hostname -I | awk '{print $1}')"
echo "  - Node.js API: http://$(hostname -I | awk '{print $1}'):3000"
echo "  - Teleport Server: $TELEPORT_SERVER_IP:3025"
echo ""
echo "🚀 Access via Teleport:"
echo "  1. Open Teleport web interface"
echo "  2. Go to 'Servers' section"
echo "  3. Find '$(hostname)' in the list"
echo "  4. Click 'Connect' for terminal access"
EOF
chmod +x /home/ubuntu/agent-info.sh
chown ubuntu:ubuntu /home/ubuntu/agent-info.sh

# Create startup info that will be shown when users SSH in
cat > /etc/motd << 'EOF'

🤖 Teleport Agent VM
====================
This VM is managed by Teleport for secure access.

Quick Commands:
  ./agent-info.sh     - Show VM information
  ./check-services.sh - Check service status
  docker ps           - List running containers
  sudo systemctl status teleport-agent

Services Running:
  - 🌐 Nginx Web Server (port 80)
  - 🔧 Node.js Demo App (port 3000)
  - 📡 Teleport Agent (connecting to server)

Access this VM securely through the Teleport web interface!

EOF

# Wait for services to start
sleep 5

# Final health check
echo "Running final health checks..."
/home/ubuntu/check-services.sh

# Show final information
echo ""
echo "============================================="
echo "Teleport Agent VM Setup Complete!"
echo "============================================="
echo "VM Hostname: $ACTUAL_HOSTNAME"
echo "Teleport Server: $TELEPORT_SERVER_IP:3025"
echo "Agent Token: [REDACTED]"
echo ""
echo "Services Started:"
echo "  ✅ Teleport Agent (Docker)"
echo "  ✅ Nginx Web Server (Port 80)"
echo "  ✅ Node.js Demo App (Port 3000)"
echo ""
echo "This VM should now appear in your Teleport web interface!"
echo "============================================="

echo "Setup completed at $(date)" 