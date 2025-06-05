terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
  backend "s3" {
    bucket               = "tf-backend-chunkr"
    key                  = "chunkr/gcp/terraform.tfstate" # Single state file, workspace-aware
    region               = "us-west-1"
    encrypt              = true
    dynamodb_table       = "tf-backend-chunkr"
    workspace_key_prefix = "workspaces" # This creates separate state files per workspace
  }
}

# Variables for secrets (will be passed via doppler run)
variable "postgres_username" {
  type        = string
  description = "PostgreSQL username from Doppler"
  sensitive   = true
}

variable "postgres_password" {
  type        = string
  description = "PostgreSQL password from Doppler"
  sensitive   = true
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

# Use locals to handle workspace-specific configurations
locals {
  workspace_configs = {
    prod = {
      project               = "lumina-prod-424120"
      base_name             = "chunkr-prod"
      region                = "us-west1"
      zone_suffix           = "b"
      chunkr_db             = "chunkr"
      keycloak_db           = "keycloak"
      gpu_vm_count          = 3
      gpu_min_vm_count      = 3
      gpu_max_vm_count      = 3
      general_vm_count      = 1
      general_min_vm_count  = 1
      general_max_vm_count  = 1
      general_machine_type  = "n2-highmem-16"
      gpu_machine_type      = "a2-highgpu-1g"
      gpu_accelerator_type  = "nvidia-tesla-a100"
      gpu_accelerator_count = 1
    }
    dev = {
      project               = "lumina-prod-424120"
      base_name             = "chunkr-dev"
      region                = "us-central1"
      zone_suffix           = "b"
      chunkr_db             = "chunkr"
      keycloak_db           = "keycloak"
      gpu_vm_count          = 1
      gpu_min_vm_count      = 1
      gpu_max_vm_count      = 1
      general_vm_count      = 1
      general_min_vm_count  = 1
      general_max_vm_count  = 1
      general_machine_type  = "n2-highmem-16"
      gpu_machine_type      = "a2-highgpu-1g"
      gpu_accelerator_type  = "nvidia-tesla-a100"
      gpu_accelerator_count = 1
    }
    staging = {
      project               = "lumina-prod-424120"
      base_name             = "chunkr-staging"
      region                = "us-east1"
      zone_suffix           = "b"
      chunkr_db             = "chunkr"
      keycloak_db           = "keycloak"
      gpu_vm_count          = 0
      gpu_min_vm_count      = 0
      gpu_max_vm_count      = 0
      general_vm_count      = 0
      general_min_vm_count  = 0
      general_max_vm_count  = 0
      general_machine_type  = "n2-highmem-8"
      gpu_machine_type      = "a2-highgpu-1g"
      gpu_accelerator_type  = "nvidia-tesla-a100"
      gpu_accelerator_count = 1
    }
    turbolearn = {
      project               = "turbolearn-ai"
      base_name             = "turbolearn-chunkr"
      region                = "us-west1"
      zone_suffix           = "b"
      chunkr_db             = "chunkr"
      keycloak_db           = "keycloak"
      gpu_vm_count          = 3
      gpu_min_vm_count      = 3
      gpu_max_vm_count      = 3
      general_vm_count      = 1
      general_min_vm_count  = 1
      general_max_vm_count  = 1
      general_machine_type  = "n2-highmem-32"
      gpu_machine_type      = "a2-highgpu-1g"
      gpu_accelerator_type  = "nvidia-tesla-a100"
      gpu_accelerator_count = 1
    }
  }

  # Get current workspace config with optional overrides
  base_config = local.workspace_configs[terraform.workspace]
  current_config = {
    project               = local.base_config.project
    base_name             = local.base_config.base_name
    region                = var.override_region != "" ? var.override_region : local.base_config.region
    zone_suffix           = var.override_zone_suffix != "" ? var.override_zone_suffix : local.base_config.zone_suffix
    chunkr_db             = local.base_config.chunkr_db
    keycloak_db           = local.base_config.keycloak_db
    gpu_vm_count          = local.base_config.gpu_vm_count
    gpu_min_vm_count      = local.base_config.gpu_min_vm_count
    gpu_max_vm_count      = local.base_config.gpu_max_vm_count
    general_vm_count      = local.base_config.general_vm_count
    general_min_vm_count  = local.base_config.general_min_vm_count
    general_max_vm_count  = local.base_config.general_max_vm_count
    general_machine_type  = local.base_config.general_machine_type
    gpu_machine_type      = local.base_config.gpu_machine_type
    gpu_accelerator_type  = local.base_config.gpu_accelerator_type
    gpu_accelerator_count = local.base_config.gpu_accelerator_count
  }
}

provider "google" {
  region  = local.current_config.region
  project = local.current_config.project
}

###############################################################
# Google Cloud Storage
###############################################################
resource "google_storage_bucket" "project_bucket" {
  name          = "${local.current_config.base_name}-bucket"
  location      = local.current_config.region
  force_destroy = true
  storage_class = "STANDARD"

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

###############################################################
# Audit Logging (Compliance Requirement)
###############################################################
resource "google_storage_bucket" "audit_logs_bucket" {
  name          = "${local.current_config.base_name}-audit-logs"
  location      = local.current_config.region
  force_destroy = true
  storage_class = "STANDARD"
}

resource "google_logging_project_sink" "audit_sink" {
  name        = "${local.current_config.base_name}-audit-sink"
  destination = "storage.googleapis.com/${google_storage_bucket.audit_logs_bucket.name}"

  unique_writer_identity = true
}

resource "google_storage_bucket_iam_member" "audit_sink_writer" {
  bucket = google_storage_bucket.audit_logs_bucket.name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.audit_sink.writer_identity
}

###############################################################
# GCS Interoperability (S3-compatible) Setup
###############################################################
resource "google_service_account" "gcs_interop" {
  account_id   = "${local.current_config.base_name}-gcs-interop"
  display_name = "GCS Interoperability Service Account"
}

resource "google_storage_hmac_key" "gcs_interop_key" {
  service_account_email = google_service_account.gcs_interop.email
}

# Grant the service account the necessary permissions to access the bucket
resource "google_storage_bucket_iam_member" "gcs_interop_object_admin" {
  bucket = google_storage_bucket.project_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.gcs_interop.email}"
}

###############################################################
# Set up the Networking Components
###############################################################
resource "google_compute_network" "vpc_network" {
  name                    = "${local.current_config.base_name}-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vpc_subnet" {
  name          = "${local.current_config.base_name}-vpc-subnet"
  ip_cidr_range = "10.3.0.0/16"
  region        = local.current_config.region
  network       = google_compute_network.vpc_network.id

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
    metadata_fields      = []
    filter_expr          = "true"
  }
}

resource "google_compute_firewall" "allow_port_8000" {
  name    = "${local.current_config.base_name}-allow-port-8000"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["port-8000-allowed"]
}

resource "google_compute_firewall" "allow_egress" {
  name    = "${local.current_config.base_name}-allow-egress"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "all"
  }

  direction          = "EGRESS"
  destination_ranges = ["0.0.0.0/0"]
}

resource "google_compute_router" "router" {
  project = local.current_config.project
  name    = "${local.current_config.base_name}-router"
  region  = local.current_config.region
  network = google_compute_network.vpc_network.id
}

module "cloud-nat" {
  source                             = "terraform-google-modules/cloud-nat/google"
  version                            = "~> 5.0"
  project_id                         = local.current_config.project
  region                             = local.current_config.region
  router                             = google_compute_router.router.name
  name                               = "${local.current_config.base_name}-nat"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${local.current_config.base_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

# Enable the Service Networking API
resource "google_project_service" "servicenetworking" {
  project            = local.current_config.project
  service            = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

# Force creation of the service networking service account and grant permissions
resource "google_project_iam_member" "servicenetworking_role" {
  project    = local.current_config.project
  role       = "roles/servicenetworking.serviceAgent"
  member     = "serviceAccount:service-${data.google_project.current.number}@service-networking.iam.gserviceaccount.com"
  depends_on = [google_project_service.servicenetworking]
}

# Get the current project data
data "google_project" "current" {
  project_id = local.current_config.project
}

resource "google_service_networking_connection" "private_service_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
  depends_on = [
    google_project_service.servicenetworking,
    google_project_iam_member.servicenetworking_role,
    google_compute_global_address.private_ip_address,
    google_compute_network.vpc_network
  ]
}

###############################################################
# K8s configuration
###############################################################
resource "google_container_cluster" "cluster" {
  name                      = "${local.current_config.base_name}-cluster"
  location                  = "${local.current_config.region}-${local.current_config.zone_suffix}"
  remove_default_node_pool  = true
  initial_node_count        = 1
  default_max_pods_per_node = 256
  deletion_protection       = false

  network    = google_compute_network.vpc_network.self_link
  subnetwork = google_compute_subnetwork.vpc_subnet.self_link

  vertical_pod_autoscaling {
    enabled = true
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "/16"
    services_ipv4_cidr_block = "/22"
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

}

resource "google_container_node_pool" "general_purpose_nodes" {
  name       = "general-compute"
  location   = "${local.current_config.region}-${local.current_config.zone_suffix}"
  cluster    = google_container_cluster.cluster.name
  node_count = local.current_config.general_vm_count

  autoscaling {
    min_node_count = local.current_config.general_min_vm_count
    max_node_count = local.current_config.general_max_vm_count
  }

  node_config {
    preemptible  = false
    machine_type = local.current_config.general_machine_type

    resource_labels = {
      "goog-gke-node-pool-provisioning-model" = "on-demand"
    }

    gcfs_config {
      enabled = true
    }

    gvnic {
      enabled = true
    }

    workload_metadata_config {
      mode = "GCE_METADATA"
    }

    labels = {
      cluster_name = "${local.current_config.base_name}-cluster"
      purpose      = "general-compute"
      node_pool    = "general-compute"
    }


    kubelet_config {
      cpu_manager_policy = "static"
      cpu_cfs_quota      = true
      pod_pids_limit     = 4096
    }

    tags = ["gke-${local.current_config.project}-${local.current_config.region}", "gke-${local.current_config.project}-${local.current_config.region}-general-compute"]
  }
}

resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-compute"
  location   = "${local.current_config.region}-${local.current_config.zone_suffix}"
  cluster    = google_container_cluster.cluster.name
  node_count = local.current_config.gpu_vm_count

  timeouts {
    create = "2h"
    update = "2h"
    delete = "2h"
  }

  autoscaling {
    min_node_count = local.current_config.gpu_min_vm_count
    max_node_count = local.current_config.gpu_max_vm_count
  }

  node_config {
    preemptible  = false
    machine_type = local.current_config.gpu_machine_type
    disk_size_gb = 500

    resource_labels = {
      "goog-gke-accelerator-type"             = "nvidia-tesla-a100"
      "goog-gke-node-pool-provisioning-model" = "on-demand"
    }

    gcfs_config {
      enabled = true
    }

    gvnic {
      enabled = true
    }

    guest_accelerator {
      type  = local.current_config.gpu_accelerator_type
      count = local.current_config.gpu_accelerator_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
      gpu_sharing_config {
        gpu_sharing_strategy       = "TIME_SHARING"
        max_shared_clients_per_gpu = 48
      }
    }

    workload_metadata_config {
      mode = "GCE_METADATA"
    }

    labels = {
      cluster_name = "${local.current_config.base_name}-cluster"
      purpose      = "gpu-time-sharing"
      node_pool    = "gpu-time-sharing"
    }

    taint {
      effect = "NO_SCHEDULE"
      key    = "nvidia.com/gpu"
      value  = "present"
    }

    kubelet_config {
      cpu_manager_policy = "static"
      cpu_cfs_quota      = true
      pod_pids_limit     = 4096
    }

    tags = ["gke-${local.current_config.project}-${local.current_config.region}", "gke-${local.current_config.project}-${local.current_config.region}-gpu-time-sharing"]
  }
}


###############################################################
# PostgreSQL (Cloud SQL)
###############################################################
resource "google_sql_database_instance" "postgres" {
  name             = "${local.current_config.base_name}-postgres"
  database_version = "POSTGRES_14"
  region           = local.current_config.region

  depends_on = [google_service_networking_connection.private_service_connection]

  settings {
    tier = "db-custom-2-4096"

    database_flags {
      name  = "max_connections"
      value = 3000
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc_network.id
    }

    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  deletion_protection = false

  timeouts {
    create = "20m"
    update = "20m"
    delete = "20m"
  }
}

resource "time_sleep" "wait_30_seconds" {
  depends_on      = [google_sql_database_instance.postgres]
  create_duration = "30s"
}

resource "google_sql_database" "chunkr-database" {
  name     = local.current_config.chunkr_db
  instance = google_sql_database_instance.postgres.name

  depends_on = [google_sql_database_instance.postgres]
}

resource "google_sql_database" "keycloak-database" {
  name     = local.current_config.keycloak_db
  instance = google_sql_database_instance.postgres.name

  depends_on = [google_sql_database_instance.postgres]
}

resource "google_sql_user" "users" {
  name     = var.postgres_username
  password = var.postgres_password
  instance = google_sql_database_instance.postgres.name

  depends_on = [
    time_sleep.wait_30_seconds,
    google_sql_database.chunkr-database,
    google_sql_database.keycloak-database
  ]
}

###############################################################
# Outputs
###############################################################
output "cluster_name" {
  value       = google_container_cluster.cluster.name
  description = "The name of the GKE cluster"
}

output "cluster_region" {
  value       = google_container_cluster.cluster.location
  description = "The region of the GKE cluster"
}

output "gke_connection_command" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.cluster.name} --region ${google_container_cluster.cluster.location}"
  description = "Command to configure kubectl to connect to the GKE cluster"
}

output "postgres_username" {
  value       = var.postgres_username
  description = "The password for the PostgreSQL database"
  sensitive   = true
}

output "postgres_password" {
  value       = var.postgres_password
  description = "The password for the PostgreSQL database"
  sensitive   = true
}

output "chunkr_postgresql_url" {
  value       = "postgresql://${var.postgres_username}:${var.postgres_password}@${google_sql_database_instance.postgres.private_ip_address}:5432/${local.current_config.chunkr_db}"
  description = "The connection URL for the PostgreSQL database"
  sensitive   = true
}

output "keycloak_postgresql_url" {
  value       = "jdbc:postgresql://${google_sql_database_instance.postgres.private_ip_address}:5432/${local.current_config.keycloak_db}"
  description = "The connection URL for the Keycloak database"
}

output "gcs_s3_compatible_endpoint" {
  value       = "https://storage.googleapis.com"
  description = "The S3-compatible endpoint for GCS"
}

output "gcs_interop_access_key" {
  value       = google_storage_hmac_key.gcs_interop_key.access_id
  description = "The access key ID for GCS interoperability (equivalent to AWS access key)"
  sensitive   = true
}

output "gcs_interop_secret_key" {
  value       = google_storage_hmac_key.gcs_interop_key.secret
  description = "The secret access key for GCS interoperability (equivalent to AWS secret key)"
  sensitive   = true
}

output "gcs_s3_compatible_region" {
  value       = "auto"
  description = "A dummy region for S3 compatibility (GCS uses a single global endpoint)"
}

output "bucket_name" {
  value       = google_storage_bucket.project_bucket.name
  description = "The name of the GCS bucket"
}
