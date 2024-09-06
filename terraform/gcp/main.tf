terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.30.0"
    }
  }
}


variable "base_name" {
  default = "chunkmydocs"
}

variable "region" {
  default = "us-central1"
}

variable "cluster_name" {
  default = "chunkmydocs-cluster"
}

variable "project" {
  type        = string
  description = "The GCP project ID"
}

variable "postgres_username" {
  type = string
  description = "The username for the PostgreSQL database"
}

variable "postgres_password" {
  type = string
  description = "The password for the PostgreSQL database"
}

variable "chunkmydocs_db" {
  default = "chunkmydocs"
}

variable "keycloak_db" {
  default = "keycloak"
}

provider "google" {
  region  = var.region
  project = var.project
}

###############################################################
# Google Cloud Storage
###############################################################
resource "google_storage_bucket" "project_bucket" {
  name          = var.base_name
  location      = var.region
  force_destroy = true
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
}

###############################################################
# Set up the Networking Components
###############################################################
# Set GOOGLE_CREDENTIALS

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
# K8s configuration
###############################################################
resource "google_container_cluster" "cluster" {
  name     = var.cluster_name
  location = "${var.region}-b"
  remove_default_node_pool = true
  initial_node_count       = 1

  deletion_protection = false

  vertical_pod_autoscaling {
    enabled = true
  }
}

resource "google_container_node_pool" "general_purpose_nodes" {
  name       = "general-compute"
  location   = "${var.region}-b"
  cluster    = google_container_cluster.cluster.name
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 6
  }

  node_config {
    preemptible  = false
    machine_type = "c2d-highcpu-4"

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
      cluster_name = var.cluster_name
      purpose      = "general-compute"
      node_pool    = "general-compute"
    }

    tags = ["gke-${var.project}-${var.region}", "gke-${var.project}-${var.region}-general-compute"]
  }
}

resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-compute"
  location   = "${var.region}-b"
  cluster    = google_container_cluster.cluster.name
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 6
  }

  node_config {
    preemptible  = false
    machine_type = "g2-standard-8"

    gcfs_config {
      enabled = true
    }

    gvnic {
      enabled = true
    }

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
      gpu_sharing_config {
        gpu_sharing_strategy       = "TIME_SHARING"
        max_shared_clients_per_gpu = 20
      }
    }

    workload_metadata_config {
      mode = "GCE_METADATA"
    }

    labels = {
      cluster_name = var.cluster_name
      purpose      = "gpu-time-sharing"
      node_pool    = "gpu-time-sharing"
    }

    taint {
      effect = "NO_SCHEDULE"
      key    = "nvidia.com/gpu"
      value  = "present"
    }

    tags = ["gke-${var.project}-${var.region}", "gke-${var.project}-${var.region}-gpu-time-sharing"]
  }
}

###############################################################
# Redis (Cloud Memorystore)
###############################################################
resource "google_redis_instance" "cache" {
  name           = "${var.base_name}-redis"
  tier           = "BASIC"
  memory_size_gb = 6

  region = var.region

  authorized_network = google_compute_network.vpc_network.id

  connect_mode = "PRIVATE_SERVICE_ACCESS"

  display_name = "${var.base_name} redis cache"

  depends_on = [google_service_networking_connection.private_service_connection]
}

# Add these new resources
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.base_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_service_networking_connection" "private_service_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

###############################################################
# PostgreSQL (Cloud SQL)
###############################################################
resource "google_sql_database_instance" "postgres" {
  name             = "${var.base_name}-postgres"
  database_version = "POSTGRES_14"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      ipv4_enabled    = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "chunkkmydocs-database" {
  name     = "${var.chunkmydocs_db}"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_database" "keycloak-database" {
  name     = "${var.keycloak_db}"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "users" {
  name     = "${var.postgres_username}"
  instance = google_sql_database_instance.postgres.name
  password = var.postgres_password
}
