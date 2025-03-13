terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
  backend "s3" {}
}

variable "project" {
  type        = string
  description = "The GCP project ID"
}

variable "region" {
  default = "us-central1"
}

variable "zone_suffix" {
  type        = string
  description = "The zone suffix to use within the region (e.g., 'a', 'b', 'c')"
  default     = "b"
}

variable "base_name" {
  type = string
}

variable "postgres_username" {
  type        = string
  description = "The username for the PostgreSQL database"
  default     = "postgres"
}

variable "postgres_password" {
  type        = string
  description = "The password for the PostgreSQL database"
  sensitive   = true
  default     = "postgres"
}

variable "chunkr_db" {
  default = "chunkr"
}

variable "keycloak_db" {
  default = "keycloak"
}

variable "general_vm_count" {
  default = 1
}

variable "general_min_vm_count" {
  default = 1
}

variable "general_max_vm_count" {
  default = 1
}

variable "general_machine_type" {
  default = "c2d-highcpu-16"
}

variable "gpu_vm_count" {
  default = 1
}

variable "gpu_min_vm_count" {
  default = 1
}

variable "gpu_max_vm_count" {
  default = 1
}

variable "gpu_machine_type" {
  default = "a2-highgpu-1g"
}

variable "gpu_accelerator_type" {
  default = "nvidia-tesla-a100"
}

variable "gpu_accelerator_count" {
  default = 1
}

provider "google" {
  region  = var.region
  project = var.project
}


###############################################################
# Google Cloud Storage
###############################################################
resource "google_storage_bucket" "project_bucket" {
  name          = "${var.base_name}-bucket"
  location      = var.region
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
# GCS Interoperability (S3-compatible) Setup
###############################################################
resource "google_service_account" "gcs_interop" {
  account_id   = "${var.base_name}-gcs-interop"
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
  name                    = "${var.base_name}-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vpc_subnet" {
  name          = "${var.base_name}-vpc-subnet"
  ip_cidr_range = "10.3.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc_network.id
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

resource "google_compute_firewall" "allow_port_8000" {
  name    = "${var.base_name}-allow-port-8000"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["port-8000-allowed"]
}

resource "google_compute_firewall" "allow_egress" {
  name    = "${var.base_name}-allow-egress"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "all"
  }

  direction          = "EGRESS"
  destination_ranges = ["0.0.0.0/0"]
}

resource "google_compute_router" "router" {
  project = var.project
  name    = "${var.base_name}-router"
  region  = var.region
  network = google_compute_network.vpc_network.id
}

module "cloud-nat" {
  source                             = "terraform-google-modules/cloud-nat/google"
  version                            = "~> 5.0"
  project_id                         = var.project
  region                             = var.region
  router                             = google_compute_router.router.name
  name                               = "${var.base_name}-nat"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.base_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_project_service" "servicenetworking" {
  project            = var.project
  service            = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

resource "google_service_networking_connection" "private_service_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
  depends_on              = [google_project_service.servicenetworking]
}
###############################################################
# K8s configuration
###############################################################
resource "google_container_cluster" "cluster" {
  name                      = "${var.base_name}-cluster"
  location                  = "${var.region}-${var.zone_suffix}"
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
  location   = "${var.region}-${var.zone_suffix}"
  cluster    = google_container_cluster.cluster.name
  node_count = var.general_vm_count

  autoscaling {
    min_node_count = var.general_min_vm_count
    max_node_count = var.general_max_vm_count
  }

  node_config {
    preemptible  = false
    machine_type = var.general_machine_type

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
      cluster_name = "${var.base_name}-cluster"
      purpose      = "general-compute"
      node_pool    = "general-compute"
    }


    kubelet_config {
      cpu_manager_policy = "static"
      cpu_cfs_quota      = true
      pod_pids_limit     = 4096
    }

    tags = ["gke-${var.project}-${var.region}", "gke-${var.project}-${var.region}-general-compute"]
  }
}

resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-compute"
  location   = "${var.region}-${var.zone_suffix}"
  cluster    = google_container_cluster.cluster.name
  node_count = var.gpu_vm_count

  timeouts {
    create = "2h"
    update = "2h"
    delete = "2h"
  }

  autoscaling {
    min_node_count = var.gpu_min_vm_count
    max_node_count = var.gpu_max_vm_count
  }

  node_config {
    preemptible  = false
    machine_type = var.gpu_machine_type
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
      type  = var.gpu_accelerator_type
      count = var.gpu_accelerator_count
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
      cluster_name = "${var.base_name}-cluster"
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

    tags = ["gke-${var.project}-${var.region}", "gke-${var.project}-${var.region}-gpu-time-sharing"]
  }
}


###############################################################
# PostgreSQL (Cloud SQL)
###############################################################
resource "google_sql_database_instance" "postgres" {
  name             = "${var.base_name}-postgres"
  database_version = "POSTGRES_14"
  region           = var.region

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
}

resource "time_sleep" "wait_30_seconds" {
  depends_on      = [google_sql_database_instance.postgres]
  create_duration = "30s"
}

resource "google_sql_database" "chunkr-database" {
  name     = var.chunkr_db
  instance = google_sql_database_instance.postgres.name

  depends_on = [google_sql_database_instance.postgres]
}

resource "google_sql_database" "keycloak-database" {
  name     = var.keycloak_db
  instance = google_sql_database_instance.postgres.name

  depends_on = [google_sql_database_instance.postgres]
}

resource "google_sql_user" "users" {
  name     = var.postgres_username
  instance = google_sql_database_instance.postgres.name
  password = var.postgres_password

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
  value       = "postgresql://${var.postgres_username}:${var.postgres_password}@${google_sql_database_instance.postgres.private_ip_address}:5432/${var.chunkr_db}"
  description = "The connection URL for the PostgreSQL database"
  sensitive   = true
}

output "keycloak_postgresql_url" {
  value       = "jdbc:postgresql://${google_sql_database_instance.postgres.private_ip_address}:5432/${var.keycloak_db}"
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
