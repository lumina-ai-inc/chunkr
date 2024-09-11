terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
}

variable "project" {
  type        = string
  description = "The GCP project ID"
}

variable "region" {
  default = "us-central1"
}

variable "base_name" {
  default     = "dev"
  description = "Base name"
}

variable "machine_type" {
  default     = "g2-standard-4"
  description = "Instance Type of the required VM"
}

variable "accelerator_type" {
  default     = "nvidia-l4"
  description = "Accelerator Type"
}

provider "google" {
  region  = var.region
  project = var.project
}

###############################################################
# Set up the Networking Components
###############################################################

resource "google_compute_network" "vpc_network" {
  name                    = "${var.base_name}-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vpc_subnet" {
  name          = "${var.base_name}-vpc-subnet"
  ip_cidr_range = "10.3.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc_network.id
}

###############################################################
# VM Instance
###############################################################
resource "google_compute_address" "vm_ip" {
  name   = "${var.base_name}-vm-ip"
  region = var.region
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.base_name}-allow-ssh"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_ports" {
  name    = "${var.base_name}-allow-ports"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "8010", "8020", "8030", "8040", "8050", "8060", "8070", "8080", "8090"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["port-8000-allowed", "port-8010-allowed", "port-8020-allowed", "port-8030-allowed", "port-8040-allowed", "port-8050-allowed", "port-8060-allowed", "port-8070-allowed", "port-8080-allowed", "port-8090-allowed"]
}

resource "google_compute_instance" "vm_instance" {
  name                      = "${var.base_name}-vm"
  machine_type              = var.machine_type
  zone                      = "${var.region}-b"
  allow_stopping_for_update = true

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 256           # Increase boot disk size to 256 GB
      type  = "pd-balanced" # Use a balanced persistent disk for better performance
    }
  }

  network_interface {
    network    = google_compute_network.vpc_network.name
    subnetwork = google_compute_subnetwork.vpc_subnet.name

    access_config {
      nat_ip = google_compute_address.vm_ip.address
    }
  }

  metadata = {
    ssh-keys = "debian:${file("~/.ssh/id_rsa.pub")}"
  }
  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y redis-tools htop git

    # Install Docker
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo apt-get install python3-venv
    # Add the debian user to the docker group
    usermod -aG docker debian

    # Pull chunk-my-docs git repository
    git clone https://github.com/lumina-ai-inc/chunk-my-docs.git /home/debian/chunk-my-docs
    chown -R debian:debian /home/debian/chunk-my-docs



    # Download and install NVIDIA GPU driver
    sudo apt-get install -y build-essential dkms
    wget https://us.download.nvidia.com/tesla/535.161.07/NVIDIA-Linux-x86_64-535.161.07.run
    chmod +x NVIDIA-Linux-x86_64-535.161.07.run
    sudo sh NVIDIA-Linux-x86_64-535.161.07.run


  EOF

  tags = ["ssh-allowed", "port-8000-allowed", "port-8010-allowed", "port-8020-allowed", "port-8030-allowed", "port-8040-allowed", "port-8050-allowed", "port-8060-allowed", "port-8070-allowed", "port-8080-allowed", "port-8090-allowed"]

  deletion_protection = false

  depends_on = [google_compute_firewall.allow_ssh, google_compute_firewall.allow_ports]
}

###############################################################
# Outputs
###############################################################
output "vm_public_ip" {
  value       = google_compute_address.vm_ip.address
  description = "The public IP address of the VM instance"
}

output "vm_ssh_command" {
  value       = "ssh debian@${google_compute_address.vm_ip.address}"
  description = "The SSH command to connect to the VM instance"
}