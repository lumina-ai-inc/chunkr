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

variable "zone" {
  default = "a"
}

variable "base_name" {
  default     = "dev"
  description = "Base name"
}

variable "machine_type" {
  default     = "g2-standard-4"
  description = "Instance Type of the required VM"
}

variable "vm_image" {
  default     = "debian-cloud/debian-11"
  description = "VM Image"
}

variable "accelerator_type" {
  default     = "nvidia-l4"
  description = "Accelerator Type"
}

provider "google" {
  region  = var.region
  project = var.project
}

variable "startup_script_path" {
  default     = "./startup.sh"
  type        = string
  description = "Path to the local startup.sh file"
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
  target_tags   = ["ssh-allowed"]
}

resource "google_compute_firewall" "allow_ports" {
  name    = "${var.base_name}-allow-ports"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "8010", "8020", "8030", "8040", "8050", "8060", "8070", "8080", "8090", "3000", "5173"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["port-8000-allowed", "port-8010-allowed", "port-8020-allowed", "port-8030-allowed", "port-8040-allowed", "port-8050-allowed", "port-8060-allowed", "port-8070-allowed", "port-8080-allowed", "port-8090-allowed", "port-3000-allowed", "port-5173-allowed"]
}

resource "google_compute_instance" "vm_instance" {
  name                      = "${var.base_name}-vm"
  machine_type              = var.machine_type
  zone                      = "${var.region}-${var.zone}"
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
      image = var.vm_image
      size  = 512
      type  = "pd-balanced"
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
    ssh-keys              = "debian:${file("~/.ssh/id_rsa.pub")}"
    install-nvidia-driver = "True"
  }

  tags = ["ssh-allowed", "port-8000-allowed", "port-8010-allowed", "port-8020-allowed", "port-8030-allowed", "port-8040-allowed", "port-8050-allowed", "port-8060-allowed", "port-8070-allowed", "port-8080-allowed", "port-8090-allowed", "port-3000-allowed", "port-5173-allowed"]

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
