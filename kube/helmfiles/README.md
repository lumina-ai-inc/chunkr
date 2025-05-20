# Helmfiles for Chunkr Deployments

This directory contains [Helmfile](https://github.com/helmfile/helmfile) configurations for different Chunkr environments.

## Structure

```
helmfiles/
├── helmfile.yaml               # Root helmfile that includes all environments
├── environments/               # Environment-specific helmfiles
│   ├── sweetspot/              # Sweetspot environment
│   │   └── helmfile.yaml
│   ├── mgx/                    # MGX environment
│   │   └── helmfile.yaml  
│   ├── turbolearn/             # Turbolearn environment
│   │   └── helmfile.yaml
│   ├── chunkr-dev/             # Chunkr Dev environment
│   │   └── helmfile.yaml
│   └── chunkr-prod/            # Chunkr Production environment
│       └── helmfile.yaml
└── README.md                   # This file
```

## Usage

To deploy a specific environment:

```bash
# Install helmfile if not already installed
# On Linux:
wget https://github.com/helmfile/helmfile/releases/download/v1.1.0/helmfile_1.1.0_linux_amd64.tar.gz
tar -zxvf helmfile_1.1.0_linux_amd64.tar.gz
chmod +x helmfile
sudo mv helmfile /usr/local/bin/

# Install the helm-diff plugin
helm plugin install https://github.com/databus23/helm-diff

# Navigate to the directory
cd kube/helmfiles

# Deploy all environments
helmfile apply

# Deploy a specific environment from /kube dir
helmfile -f helmfiles/environments/sweetspot/helmfile.yaml apply
```

Each environment's helmfile is configured with the appropriate chart values and settings based on the original deployment scripts.
