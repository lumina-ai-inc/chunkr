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
  type        = string
  description = "The username for the PostgreSQL database"
}

variable "postgres_password" {
  type        = string
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

###############################################################
# K8s configuration
###############################################################
resource "google_container_cluster" "cluster" {
  name                     = var.cluster_name
  location                 = "${var.region}-b"
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
    disk_size_gb = 1000

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

  transit_encryption_mode = "DISABLED"

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

resource "google_project_service" "servicenetworking" {
  project = var.project
  service = "servicenetworking.googleapis.com"

  disable_on_destroy = false
}

resource "google_service_networking_connection" "private_service_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
  depends_on              = [google_project_service.servicenetworking]
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
      ipv4_enabled = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "chunkkmydocs-database" {
  name     = var.chunkmydocs_db
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_database" "keycloak-database" {
  name     = var.keycloak_db
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "users" {
  name     = var.postgres_username
  instance = google_sql_database_instance.postgres.name
  password = var.postgres_password
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

resource "google_compute_instance" "vm_instance" {
  name         = "${var.base_name}-vm"
  machine_type = "e2-standard-2"
  zone         = "${var.region}-b"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
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
    apt-get install -y redis-tools htop

    # Install Docker
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add the debian user to the docker group
    usermod -aG docker debian

    # Wait for the Docker Compose file to be copied
    while [ ! -f /home/debian/docker-compose.yml ]; do
      sleep 5
    done

    # Start Docker Compose services
    cd /home/debian
    docker compose up -d
  EOF

  tags = ["ssh-allowed", "port-8000-allowed"]

  provisioner "file" {
    source      = "./compose.yaml"
    destination = "/home/debian/compose.yaml"

    connection {
      type        = "ssh"
      user        = "debian"
      private_key = file("~/.ssh/id_rsa")
      host        = self.network_interface[0].access_config[0].nat_ip
    }
  }

  depends_on = [google_compute_firewall.allow_ssh]
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

output "chunkmydocs_postgresql_url" {
  value       = "postgresql://${var.postgres_username}:${var.postgres_password}@${google_sql_database_instance.postgres.public_ip_address}:5432/${var.chunkmydocs_db}"
  description = "The connection URL for the PostgreSQL database"
  sensitive   = true
}

output "keycloak_postgresql_url" {
  value       = "postgresql://${google_sql_database_instance.postgres.public_ip_address}:5432/${var.keycloak_db}"
  description = "The connection URL for the Keycloak database"
}

output "redis_url" {
  value       = "redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
  description = "The connection URL for the Redis cache"
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

output "vm_public_ip" {
  value       = google_compute_address.vm_ip.address
  description = "The public IP address of the VM instance"
}
