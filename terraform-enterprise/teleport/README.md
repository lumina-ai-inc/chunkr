# Teleport Infrastructure Setup

Complete SOC2-compliant Teleport setup on Google Cloud Platform with server and agent VMs. Access via **Google IAP** (Identity-Aware Proxy) - no SSH ports exposed to internet!

## Architecture

```
Your Browser → Teleport Server (GCP) → Agent VM 1 (with demo services)
                                    → Agent VM 2 (with demo services)
                                    → Agent VM 3 (with demo services)
```

## What You Get

- **🖥️ Teleport Server** with web interface at `https://SERVER_IP:3080`
- **🤖 Agent VMs** (3 by default) with auto-connecting Docker agents
- **🌐 Demo Services** on each agent (Nginx + Node.js API)
- **🔒 SOC2 Compliance** - no SSH ports, encrypted connections, session recording
- **📊 Management Tools** for deployment, scaling, and monitoring

## Prerequisites

1. **GCP CLI** installed and authenticated: `gcloud auth login`
2. **IAP permissions** in your GCP project:
   ```bash
   # Grant yourself IAP access
   gcloud projects add-iam-policy-binding lumina-prod-424120 \
     --member=user:your-email@domain.com \
     --role=roles/iap.tunnelResourceAccessor
   
   # Grant OS Login access
   gcloud projects add-iam-policy-binding lumina-prod-424120 \
     --member=user:your-email@domain.com \
     --role=roles/compute.osLogin
   ```

## Quick Start

### 1. Deploy Teleport Server

```bash
cd gcp/client
./tf-teleport.sh init
./tf-teleport.sh apply
```

### 2. Create Admin User

```bash
# Access server via IAP (SOC2 compliant - no SSH ports!)
gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap

# Create admin user
./create-admin-user.sh your-username
# Save the signup link!
```

### 3. Deploy Agent VMs

```bash
cd ../agent

# Get server IP
SERVER_IP=$(cd ../client && ./tf-teleport.sh output | grep teleport_server_ip | cut -d'"' -f4)

# Access server via IAP and create token
# gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap
# ./create-token.sh
# Copy the token, then:

./tf-agent.sh init
./tf-agent.sh apply --teleport-server-ip $SERVER_IP --teleport-token YOUR_TOKEN
```

### 4. Access Everything

```bash
# Check agent status
./tf-agent.sh check-agents

# Open web interface: https://SERVER_IP:3080
# Go to "Servers" to see all VMs
```

## Configuration Options

### Server Configuration
- **Project:** lumina-prod-424120
- **Default:** us-central1-b, n2-standard-2, 50GB SSD
- **Override:** `./tf-teleport.sh apply --override-region us-west1 --override-zone c`

### Agent Configuration
- **Default:** 3 agents, n2-standard-2, 30GB SSD
- **Custom:** `./tf-agent.sh apply --teleport-server-ip $IP --teleport-token $TOKEN --agent-count 5 --machine-type n2-standard-4`

## Management Commands

```bash
# Server Management
cd gcp/client
./tf-teleport.sh status          # Check server status
./tf-teleport.sh output          # View server details

# Agent Management  
cd gcp/agent
./tf-agent.sh check-agents       # Check all agents
./tf-agent.sh status             # View agent infrastructure
./tf-agent.sh output             # View agent details

# Scaling
./tf-agent.sh apply --teleport-server-ip $IP --teleport-token $TOKEN --agent-count 10
```

## Agent VM Features

Each agent VM includes:
- **Teleport Agent** (Docker) - auto-connects to server
- **Nginx Web Server** (port 80) - demo site with VM info
- **Node.js API** (port 3000) - JSON endpoints with system stats
- **Health Checks** - `./agent-info.sh` and `./check-services.sh`

## Security & Compliance

**🔒 SOC2 Compliant Architecture:**
- ✅ **No SSH ports** exposed to internet (IAP tunnel only)
- ✅ **Google IAP** for secure VM access with identity verification
- ✅ **OS Login** with centralized user management
- ✅ **Encrypted connections** (TLS) between all components
- ✅ **Session recording** and comprehensive audit logs
- ✅ **Agent-based** connections (outbound only)
- ✅ **Zero-trust network** with firewall restrictions

**🛡️ Identity-Aware Proxy (IAP) Benefits:**
- **No VPN needed** - Access through Google's secure proxy
- **Identity verification** - All access tied to Google accounts
- **Automatic logging** - Every connection is audited
- **Conditional access** - Can restrict based on device/location
- **No exposed ports** - SSH only available via IAP tunnel

## Accessing VMs

1. **Open Teleport:** `https://SERVER_IP:3080`
2. **Login** with admin credentials
3. **Navigate to "Servers"** - see all connected VMs
4. **Click "Connect"** for instant terminal access
5. **Use port forwarding** to access services (ports 80, 3000)

## Manual Agent Setup

To connect additional VMs manually (these VMs don't need SSH access - agents connect outbound only):

```bash
# 1. Get token from server (via IAP)
gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap
./create-token.sh

# 2. Run on target VM (can be anywhere, no SSH ports needed)
docker run -d \
  --name teleport-agent \
  --hostname $(hostname) \
  --net=host \
  --pid=host \
  -v /etc/machine-id:/etc/machine-id:ro \
  -v /var/lib/teleport:/var/lib/teleport \
  public.ecr.aws/gravitational/teleport:13 \
  teleport start \
    --roles=node \
    --token=YOUR_TOKEN \
    --auth-server=YOUR_TELEPORT_IP:3025 \
    --labels=env=production,hostname=$(hostname)
```

**Note:** Target VMs only need outbound HTTPS/port 3025 access - no inbound SSH required!

## Troubleshooting

```bash
# Check server logs (via IAP)
gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap
sudo tail -f /var/log/teleport-setup.log

# Check agent logs (via IAP)
gcloud compute ssh teleport-agent-1 --zone us-central1-b --tunnel-through-iap
sudo docker logs teleport-agent

# Test connectivity from agent to server
nc -z SERVER_IP 3025

# Restart services
sudo docker restart teleport-agent
```

## Files Structure

```
gcp/
├── client/                  # Teleport server
│   ├── main.tf             # Server infrastructure
│   ├── startup-script.sh   # Server setup
│   └── tf-teleport.sh      # Server management
└── agent/                   # Agent VMs
    ├── main.tf             # Agent infrastructure  
    ├── startup-script.sh   # Agent setup
    └── tf-agent.sh         # Agent management
```

## Use Cases

Perfect for:
- **SOC2 compliance** requirements
- **Secure remote access** without SSH
- **Centralized access management**
- **Session recording** and audit trails
- **Testing Teleport** before production deployment

## Clean Up

```bash
# Destroy agents first
cd gcp/agent && ./tf-agent.sh destroy

# Then destroy server
cd ../client && ./tf-teleport.sh destroy
```

---

**Next Steps:** Deploy the server, create your admin user, then deploy agents. Everything connects automatically! 