terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
  backend "s3" {
    bucket               = "tf-backend-chunkr"
    key                  = "teleport/agent/gcp/terraform.tfstate"
    region               = "us-west-1"
    encrypt              = true
    dynamodb_table       = "tf-backend-chunkr"
    workspace_key_prefix = "teleport-agent-workspaces"
  }
}

# Variables for Teleport server connection
variable "teleport_server_ip" {
  type        = string
  description = "The IP address of the Teleport server"
}

variable "teleport_token" {
  type        = string
  description = "The Teleport token for agents to connect"
  sensitive   = true
}

variable "teleport_ca_pin" {
  type        = string
  description = "The Teleport CA pin for agents to verify server identity"
  sensitive   = true
}

# Optional location override variables
variable "override_region" {
  type        = string
  description = "Override the default region for DR testing"
  default     = ""
}

variable "override_zone_suffix" {
  type        = string
  description = "Override the default zone suffix for DR testing"
  default     = ""
}

# Agent configuration
variable "agent_count" {
  type        = number
  description = "Number of agent VMs to deploy"
  default     = 1
}

variable "agent_machine_type" {
  type        = string
  description = "Machine type for agent VMs"
  default     = "n2-standard-32"
}

variable "agent_disk_size_gb" {
  type        = number
  description = "Disk size for agent VMs in GB"
  default     = 128
}

# Configuration
locals {
  config = {
    project             = "lumina-prod-424120"
    base_name           = "teleport-agent"
    region              = var.override_region != "" ? var.override_region : "us-central1"
    zone_suffix         = var.override_zone_suffix != "" ? var.override_zone_suffix : "b"
    machine_type        = var.agent_machine_type
    disk_size_gb        = var.agent_disk_size_gb
    agent_count         = var.agent_count
    allowed_cidr_blocks = ["0.0.0.0/0"] # Agents need outbound access
  }
}

provider "google" {
  region  = local.config.region
  project = local.config.project
}

###############################################################
# Data Sources
###############################################################

# Get the Teleport network from the client configuration
data "google_compute_network" "teleport_network" {
  name = "teleport-network"
}

data "google_compute_subnetwork" "teleport_subnet" {
  name   = "teleport-subnet"
  region = local.config.region
}

###############################################################
# Firewall Rules for Agent VMs
###############################################################

# Allow agents to connect to Teleport server
resource "google_compute_firewall" "agent_to_teleport" {
  name    = "${local.config.base_name}-to-teleport"
  network = data.google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["3025"] # Teleport auth server
  }

  source_tags = ["teleport-agent"]
  target_tags = ["teleport-server"]
}

# Allow IAP access to agents for administration (SOC2 compliant)
resource "google_compute_firewall" "agent_iap" {
  name    = "${local.config.base_name}-iap"
  network = data.google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"] # IAP source range
  target_tags   = ["teleport-agent"]
}

# Allow incoming connections that will be managed by Teleport
resource "google_compute_firewall" "agent_services" {
  name    = "${local.config.base_name}-services"
  network = data.google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080", "8443", "3000", "5000", "9000"]
  }

  source_ranges = ["10.4.0.0/16"] # Internal network only
  target_tags   = ["teleport-agent"]
}

# Allow Chunkr HTTPS services (for external access)
resource "google_compute_firewall" "agent_chunkr_https" {
  name    = "${local.config.base_name}-chunkr-https"
  network = data.google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["443", "8444", "8443", "9100"] # Web UI, API, Keycloak, MinIO
  }

  source_ranges = ["0.0.0.0/0"] # Allow external access for Chunkr services
  target_tags   = ["teleport-agent"]
}

###############################################################
# Service Account for Agent VMs
###############################################################

resource "google_service_account" "agent_vm" {
  account_id   = "${local.config.base_name}-vm"
  display_name = "Teleport Agent VM Service Account"
  description  = "Service account for Teleport Agent VMs"
}

# Grant necessary permissions
resource "google_project_iam_member" "agent_vm_logging" {
  project = local.config.project
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agent_vm.email}"
}

resource "google_project_iam_member" "agent_vm_monitoring" {
  project = local.config.project
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.agent_vm.email}"
}

###############################################################
# Agent VMs
###############################################################

# Startup script for agent VMs
locals {
  startup_script = templatefile("${path.module}/startup-script.sh", {
    teleport_server_ip = var.teleport_server_ip
    teleport_token     = var.teleport_token
    teleport_ca_pin    = var.teleport_ca_pin
    agent_hostname     = "agent-${local.config.region}"
    region             = local.config.region
  })
}

# Create agent VMs
resource "google_compute_instance" "agent_vms" {
  count = local.config.agent_count

  name         = "${local.config.base_name}-${count.index + 1}"
  machine_type = local.config.machine_type
  zone         = "${local.config.region}-${local.config.zone_suffix}"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = local.config.disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    network    = data.google_compute_network.teleport_network.self_link
    subnetwork = data.google_compute_subnetwork.teleport_subnet.self_link

    # Optional: Give external IP for easier administration
    access_config {
      // Ephemeral IP
    }
  }

  service_account {
    email  = google_service_account.agent_vm.email
    scopes = ["cloud-platform"]
  }

  tags = ["teleport-agent"]

  metadata = {
    "enable-oslogin"         = "TRUE"
    "block-project-ssh-keys" = "TRUE"
    "startup-script"         = local.startup_script
    "teleport-server-ip"     = var.teleport_server_ip
    "teleport-token"         = var.teleport_token
    "teleport-ca-pin"        = var.teleport_ca_pin
    "agent-id"               = "${local.config.base_name}-${count.index + 1}"
  }

  metadata_startup_script = local.startup_script

  # Allow stopping for updates
  allow_stopping_for_update = true

  # Add labels for better organization
  labels = {
    environment = "production"
    component   = "teleport-agent"
    managed_by  = "terraform"
  }
}

###############################################################
# Outputs
###############################################################

output "agent_vm_names" {
  value       = google_compute_instance.agent_vms[*].name
  description = "Names of the agent VMs"
}

output "agent_vm_internal_ips" {
  value       = google_compute_instance.agent_vms[*].network_interface[0].network_ip
  description = "Internal IP addresses of the agent VMs"
}

output "agent_vm_external_ips" {
  value       = google_compute_instance.agent_vms[*].network_interface[0].access_config[0].nat_ip
  description = "External IP addresses of the agent VMs"
}

output "agent_iap_ssh_commands" {
  value = [
    for vm in google_compute_instance.agent_vms :
    "gcloud compute ssh ${vm.name} --zone ${vm.zone} --tunnel-through-iap"
  ]
  description = "Commands to securely access agent VMs via IAP (SOC2 compliant)"
}

output "agent_count" {
  value       = local.config.agent_count
  description = "Number of agent VMs deployed"
}

output "teleport_server_connection" {
  value       = "${var.teleport_server_ip}:3025"
  description = "Teleport server connection string used by agents"
}

output "agent_status_check" {
  value = [
    for vm in google_compute_instance.agent_vms :
    "gcloud compute ssh ${vm.name} --zone ${vm.zone} --tunnel-through-iap --command 'sudo systemctl status teleport-agent'"
  ]
  description = "Commands to check agent status on each VM via IAP"
}

output "chunkr_service_urls" {
  value = [
    for vm in google_compute_instance.agent_vms :
    {
      vm_name  = vm.name
      web_ui   = "https://${vm.network_interface[0].access_config[0].nat_ip}:443"
      api      = "https://${vm.network_interface[0].access_config[0].nat_ip}:8444"
      keycloak = "https://${vm.network_interface[0].access_config[0].nat_ip}:8443"
      minio    = "https://${vm.network_interface[0].access_config[0].nat_ip}:9100"
    }
  ]
  description = "Chunkr service URLs for each agent VM (HTTPS configuration)"
}
