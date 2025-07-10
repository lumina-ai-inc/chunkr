terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
  backend "s3" {
    bucket               = "tf-backend-chunkr"
    key                  = "teleport/gcp/terraform.tfstate"
    region               = "us-west-1"
    encrypt              = true
    dynamodb_table       = "tf-backend-chunkr"
    workspace_key_prefix = "teleport-workspaces"
  }
}

# Optional location override variables for DR testing
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

# Configuration
locals {
  config = {
    project             = "lumina-prod-424120"
    base_name           = "teleport"
    region              = var.override_region != "" ? var.override_region : "us-central1"
    zone_suffix         = var.override_zone_suffix != "" ? var.override_zone_suffix : "b"
    machine_type        = "n2-standard-2"
    disk_size_gb        = 50
    allowed_cidr_blocks = ["0.0.0.0/0"] # Adjust as needed for your security requirements
  }
}

provider "google" {
  region  = local.config.region
  project = local.config.project
}

###############################################################
# Networking
###############################################################

# Create VPC network for Teleport
resource "google_compute_network" "teleport_network" {
  name                    = "${local.config.base_name}-network"
  auto_create_subnetworks = false
}

# Create subnet for Teleport
resource "google_compute_subnetwork" "teleport_subnet" {
  name          = "${local.config.base_name}-subnet"
  ip_cidr_range = "10.4.0.0/16"
  region        = local.config.region
  network       = google_compute_network.teleport_network.id

  # Enable flow logs for network monitoring and security compliance
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# Firewall rules for Teleport
resource "google_compute_firewall" "teleport_web" {
  name    = "${local.config.base_name}-web"
  network = google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["3080"] # Teleport web interface
  }

  source_ranges = local.config.allowed_cidr_blocks
  target_tags   = ["teleport-server"]
}

resource "google_compute_firewall" "teleport_auth" {
  name    = "${local.config.base_name}-auth"
  network = google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["3025"] # Teleport auth server (for agents)
  }

  source_ranges = ["0.0.0.0/0"] # Agents connect from anywhere
  target_tags   = ["teleport-server"]
}

resource "google_compute_firewall" "teleport_ssh" {
  name    = "${local.config.base_name}-ssh"
  network = google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["3022"] # Teleport SSH (for agent connections)
  }

  source_ranges = ["0.0.0.0/0"] # Agents connect from anywhere
  target_tags   = ["teleport-server"]
}

# IAP firewall rule for secure access (no SSH ports needed)
resource "google_compute_firewall" "teleport_iap" {
  name    = "${local.config.base_name}-iap"
  network = google_compute_network.teleport_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"] # Only for IAP tunneling
  }

  source_ranges = ["35.235.240.0/20"] # IAP source range
  target_tags   = ["teleport-server"]
}

###############################################################
# IAP Configuration for SOC2 Compliance
###############################################################

# Enable IAP for the project
resource "google_project_service" "iap" {
  project = local.config.project
  service = "iap.googleapis.com"
}

# IAM bindings for IAP access
resource "google_project_iam_member" "iap_tunnel_user" {
  project = local.config.project
  role    = "roles/iap.tunnelResourceAccessor"
  member  = "serviceAccount:${google_service_account.teleport_vm.email}"
}

# Allow the VM service account to use OS Login
resource "google_project_iam_member" "teleport_vm_oslogin" {
  project = local.config.project
  role    = "roles/compute.osLogin"
  member  = "serviceAccount:${google_service_account.teleport_vm.email}"
}

resource "google_compute_firewall" "teleport_egress" {
  name    = "${local.config.base_name}-egress"
  network = google_compute_network.teleport_network.name

  allow {
    protocol = "all"
  }

  direction          = "EGRESS"
  destination_ranges = ["0.0.0.0/0"]
}

# NAT router for outbound internet access
resource "google_compute_router" "teleport_router" {
  name    = "${local.config.base_name}-router"
  region  = local.config.region
  network = google_compute_network.teleport_network.id
}

module "teleport_cloud_nat" {
  source  = "terraform-google-modules/cloud-nat/google"
  version = "~> 5.0"

  project_id                         = local.config.project
  region                             = local.config.region
  router                             = google_compute_router.teleport_router.name
  name                               = "${local.config.base_name}-nat"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

###############################################################
# Service Account for Teleport VM
###############################################################

resource "google_service_account" "teleport_vm" {
  account_id   = "${local.config.base_name}-vm"
  display_name = "Teleport VM Service Account"
  description  = "Service account for Teleport VM"
}

# Grant necessary permissions
resource "google_project_iam_member" "teleport_vm_logging" {
  project = local.config.project
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.teleport_vm.email}"
}

resource "google_project_iam_member" "teleport_vm_monitoring" {
  project = local.config.project
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.teleport_vm.email}"
}

###############################################################
# Teleport VM
###############################################################

# Create static IP for Teleport server
resource "google_compute_address" "teleport_ip" {
  name   = "${local.config.base_name}-ip"
  region = local.config.region
}

# Startup script for Teleport VM
locals {
  startup_script = templatefile("${path.module}/startup-script.sh", {
    teleport_hostname = google_compute_address.teleport_ip.address
  })
}

# Create Teleport VM
resource "google_compute_instance" "teleport_server" {
  name         = "${local.config.base_name}-server"
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
    network    = google_compute_network.teleport_network.self_link
    subnetwork = google_compute_subnetwork.teleport_subnet.self_link

    access_config {
      nat_ip = google_compute_address.teleport_ip.address
    }
  }

  service_account {
    email  = google_service_account.teleport_vm.email
    scopes = ["cloud-platform"]
  }

  tags = ["teleport-server"]

  metadata = {
    "enable-oslogin"         = "TRUE"
    "block-project-ssh-keys" = "TRUE"
    "startup-script"         = local.startup_script
  }

  metadata_startup_script = local.startup_script

  # Allow stopping for updates
  allow_stopping_for_update = true
}

###############################################################
# Outputs
###############################################################

output "teleport_server_ip" {
  value       = google_compute_address.teleport_ip.address
  description = "The external IP address of the Teleport server"
}

output "teleport_web_url" {
  value       = "https://${google_compute_address.teleport_ip.address}:3080"
  description = "The URL to access the Teleport web interface"
}

output "teleport_auth_server" {
  value       = "${google_compute_address.teleport_ip.address}:3025"
  description = "The auth server address for Teleport agents"
}

output "teleport_vm_name" {
  value       = google_compute_instance.teleport_server.name
  description = "The name of the Teleport VM"
}

output "iap_ssh_command" {
  value       = "gcloud compute ssh ${google_compute_instance.teleport_server.name} --zone ${google_compute_instance.teleport_server.zone} --tunnel-through-iap"
  description = "Command to securely access the Teleport server via IAP (SOC2 compliant)"
}

output "teleport_network_name" {
  value       = google_compute_network.teleport_network.name
  description = "The name of the Teleport network"
}

output "agent_connection_command" {
  value       = "teleport start --roles=node --token=YOUR_TOKEN_HERE --auth-server=${google_compute_address.teleport_ip.address}:3025 --labels=env=production,hostname=$(hostname)"
  description = "Command to run on target VMs to connect them to Teleport (requires native teleport installation)"
}
