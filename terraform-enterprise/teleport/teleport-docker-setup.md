# Teleport Docker Setup Guide

## Overview

Teleport is an agent-based solution that provides secure access to VMs without requiring SSH ports to be open. Agents connect outbound to the Teleport server, making it perfect for SOC2 compliance and environments where SSH ports are restricted.

## Architecture

```
Your Browser → Teleport Server (Docker) → Agent on VM 1
                                       → Agent on VM 2  
                                       → Agent on VM 3
```

**Key Benefits:**
- ✅ No SSH ports needed on target VMs
- ✅ Web-based terminal access
- ✅ Session recording and audit logs
- ✅ Multi-VM management from one interface
- ✅ SOC2 compliant

## Prerequisites

- Docker and Docker Compose installed
- One server to run Teleport (minimum 2GB RAM, 20GB disk)
- Network connectivity from target VMs to Teleport server
- Target VMs must allow outbound connections on port 3025

## Part 1: Teleport Server Setup

### 1. Create Project Directory

```bash
mkdir teleport-docker
cd teleport-docker
```

### 2. Create Docker Compose File

Create `docker-compose.yml`:

```yaml
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
      - TELEPORT_HOSTNAME=your-server-hostname-or-ip
    command: ["teleport", "start", "--config=/etc/teleport/teleport.yaml"]
    restart: unless-stopped
```

### 3. Create Teleport Configuration

Create `config/teleport.yaml`:

```yaml
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
  cluster_name: "teleport-cluster"
  tokens:
    - "proxy,node:your-secret-token-here"

proxy_service:
  enabled: "yes"
  listen_addr: 0.0.0.0:3023
  web_listen_addr: 0.0.0.0:3080
  tunnel_listen_addr: 0.0.0.0:3024
  public_addr: your-server-hostname-or-ip:3080

ssh_service:
  enabled: "yes"
  listen_addr: 0.0.0.0:3022
```

### 4. Create Required Directories

```bash
mkdir -p teleport-data config
```

### 5. Start Teleport Server

```bash
# Start the server
docker-compose up -d

# Check logs
docker-compose logs -f teleport
```

### 6. Create Initial Admin User

```bash
# Create admin user
docker-compose exec teleport tctl users add admin --roles=editor,access --logins=root,ubuntu,admin

# This will output a signup link - save it!
```

## Part 2: Agent Setup on Target VMs

### Method 1: Docker Agent (Recommended)

On each VM you want to manage:

```bash
# Run Teleport agent
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
    --token=your-secret-token-here \
    --auth-server=YOUR_TELEPORT_SERVER_IP:3025 \
    --labels=env=production,team=devops
```

### Method 2: Native Agent Installation

```bash
# Download and install Teleport
curl -O https://cdn.teleport.dev/teleport-v13.4.13-linux-amd64-bin.tar.gz
tar -xzf teleport-v13.4.13-linux-amd64-bin.tar.gz
sudo ./teleport/install

# Start agent
sudo teleport start \
  --roles=node \
  --token=your-secret-token-here \
  --auth-server=YOUR_TELEPORT_SERVER_IP:3025 \
  --labels=env=production,team=devops
```

### Method 3: Agent Docker Compose

Create `docker-compose.yml` on each VM:

```yaml
version: '3.8'

services:
  teleport-agent:
    image: public.ecr.aws/gravitational/teleport:13
    container_name: teleport-agent
    hostname: ${HOSTNAME}
    network_mode: host
    pid: host
    volumes:
      - /etc/machine-id:/etc/machine-id:ro
      - /var/lib/teleport:/var/lib/teleport
    environment:
      - TELEPORT_AUTH_SERVER=YOUR_TELEPORT_SERVER_IP:3025
      - TELEPORT_TOKEN=your-secret-token-here
    command: [
      "teleport", "start",
      "--roles=node",
      "--token=${TELEPORT_TOKEN}",
      "--auth-server=${TELEPORT_AUTH_SERVER}",
      "--labels=env=production,hostname=${HOSTNAME}"
    ]
    restart: unless-stopped
```

## Part 3: Access and Usage

### 1. Access Web Interface

Open your browser and go to:
```
https://YOUR_TELEPORT_SERVER_IP:3080
```

### 2. Complete Initial Setup

1. Use the signup link from step 6 of server setup
2. Create your admin password
3. Set up two-factor authentication (recommended)

### 3. Access Your VMs

1. **Login** to the web interface
2. **Navigate** to "Servers" section
3. **See all connected VMs** with their labels
4. **Click "Connect"** on any VM
5. **Get instant terminal access** in your browser

### 4. Manage Users and Access

```bash
# Add new user
docker-compose exec teleport tctl users add newuser --roles=access --logins=ubuntu,root

# List users
docker-compose exec teleport tctl users ls

# Create custom role
docker-compose exec teleport tctl create -f custom-role.yaml
```

## Network Requirements

### Teleport Server
- **Port 3080**: Web interface (HTTPS)
- **Port 3025**: Auth server (for agents)
- **Port 3022**: SSH (optional)

### Target VMs
- **Outbound access** to Teleport server on port 3025
- **No inbound ports** required!

## Security Configuration

### 1. Generate Strong Token

```bash
# Generate secure token
openssl rand -hex 32
```

### 2. Configure HTTPS

Add SSL certificates to your configuration:

```yaml
proxy_service:
  enabled: "yes"
  web_listen_addr: 0.0.0.0:3080
  https_keypairs:
    - key_file: /etc/teleport/server.key
      cert_file: /etc/teleport/server.crt
```

### 3. Enable Session Recording

```yaml
auth_service:
  enabled: "yes"
  session_recording: "node"  # or "proxy" or "off"
```

## Troubleshooting

### Check Server Status
```bash
docker-compose logs teleport
docker-compose exec teleport tctl status
```

### Check Agent Connection
```bash
# On agent VM
docker logs teleport-agent

# On server
docker-compose exec teleport tctl nodes ls
```

### Common Issues

1. **Agent not connecting**: Check firewall rules for port 3025
2. **Token expired**: Generate new token with `tctl tokens add`
3. **DNS issues**: Use IP addresses instead of hostnames
4. **Permission denied**: Ensure user has correct roles and logins

## Backup and Recovery

### Backup Teleport Data
```bash
# Create backup
tar -czf teleport-backup-$(date +%Y%m%d).tar.gz teleport-data/

# Restore from backup
tar -xzf teleport-backup-YYYYMMDD.tar.gz
```

## Scaling

### Multiple Teleport Instances
For high availability, you can run multiple Teleport instances behind a load balancer.

### Large Number of VMs
Teleport can handle thousands of connected nodes. Consider:
- Increasing server resources
- Using node labels for organization
- Implementing proper RBAC

## Next Steps

1. **Set up proper SSL certificates**
2. **Configure user roles and permissions**
3. **Enable session recording**
4. **Set up monitoring and alerting**
5. **Implement backup procedures**

## Support

- **Documentation**: https://goteleport.com/docs/
- **Community**: https://github.com/gravitational/teleport
- **Enterprise Support**: Available for production deployments

## Configuration Examples

### Custom Role Example

Create `custom-role.yaml`:

```yaml
kind: role
version: v5
metadata:
  name: developer
spec:
  allow:
    logins: [ubuntu, developer]
    node_labels:
      env: [development, staging]
    rules:
      - resources: [session]
        verbs: [list, read]
  deny: {}
```

### Node Labels Example

Organize your VMs with labels:

```bash
# Production web servers
--labels=env=production,service=web,team=frontend

# Development databases  
--labels=env=development,service=database,team=backend

# Staging environment
--labels=env=staging,service=api,team=backend
```

This setup provides a complete, SOC2-compliant solution for accessing multiple VMs without requiring SSH ports to be open! 