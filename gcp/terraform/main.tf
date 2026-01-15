terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "orin-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Compute Engine VM for GPU workloads
resource "google_compute_instance" "orin_production" {
  name         = "orin-production"
  machine_type = "n1-standard-8"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 200
      type  = "pd-ssd"
    }
  }

  # GPU for OCR/Segmentation workloads
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral IP
    }
  }

  metadata = {
    gce-container-declaration = templatefile("${path.module}/container-manifest.yaml", {
      project_id = var.project_id
    })
  }

  service_account {
    email  = google_service_account.orin_sa.email
    scopes = ["cloud-platform"]
  }

  tags = ["orin", "http-server", "https-server"]
}

# Service Account
resource "google_service_account" "orin_sa" {
  account_id   = "orin-production"
  display_name = "Orin Production Service Account"
}

# Cloud Storage bucket for document storage
resource "google_storage_bucket" "documents" {
  name          = "${var.project_id}-orin-documents"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

# Cloud Storage bucket for backups
resource "google_storage_bucket" "backups" {
  name          = "${var.project_id}-orin-backups"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

# Load Balancer for HTTPS termination
resource "google_compute_global_address" "default" {
  name = "orin-lb-ip"
}

resource "google_compute_health_check" "default" {
  name = "orin-health-check"

  http_health_check {
    port         = 8000
    request_path = "/health"
  }

  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

resource "google_compute_backend_service" "default" {
  name          = "orin-backend"
  protocol      = "HTTP"
  port_name     = "http"
  timeout_sec   = 30
  health_checks = [google_compute_health_check.default.id]

  backend {
    group = google_compute_instance_group.default.id
  }
}

resource "google_compute_instance_group" "default" {
  name = "orin-instance-group"
  zone = "${var.region}-a"

  instances = [
    google_compute_instance.orin_production.id
  ]

  named_port {
    name = "http"
    port = "8000"
  }
}

# Firewall rules
resource "google_compute_firewall" "allow_http" {
  name    = "orin-allow-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "5173", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["orin"]
}

# Secret Manager for sensitive configuration
resource "google_secret_manager_secret" "db_password" {
  secret_id = "orin-db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "api_keys" {
  secret_id = "orin-api-keys"

  replication {
    auto {}
  }
}

# IAM bindings
resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orin_sa.email}"
}

# Optional: Cloud SQL for managed Postgres (future migration)
# Commented out for now, keeping containerized Postgres
# resource "google_sql_database_instance" "postgres" {
#   name             = "orin-postgres"
#   database_version = "POSTGRES_15"
#   region           = var.region
#
#   settings {
#     tier = "db-custom-4-16384"
#
#     ip_configuration {
#       ipv4_enabled    = false
#       private_network = google_compute_network.default.id
#     }
#   }
# }

# Optional: Cloud Memorystore for Redis (future migration)
# resource "google_redis_instance" "cache" {
#   name           = "orin-redis"
#   tier           = "STANDARD_HA"
#   memory_size_gb = 5
#   region         = var.region
# }

output "instance_ip" {
  value = google_compute_instance.orin_production.network_interface[0].access_config[0].nat_ip
}

output "load_balancer_ip" {
  value = google_compute_global_address.default.address
}

output "documents_bucket" {
  value = google_storage_bucket.documents.name
}

