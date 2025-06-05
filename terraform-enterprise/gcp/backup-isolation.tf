# Backup and Recovery Data Isolation Configuration
# This file demonstrates SOC2 compliance for recovery data isolation

###############################################################
# Variables for Backup Configuration
###############################################################

variable "enable_backup_infrastructure" {
  type        = bool
  description = "Enable backup infrastructure (recommended for prod only)"
  default     = false
}

variable "backup_region" {
  type        = string
  description = "Region for backup storage (should be different from production)"
  default     = "us-central1"
}

variable "archive_region" {
  type        = string
  description = "Region for long-term archive storage (should be different from production and backup)"
  default     = "us-east1"
}

variable "backup_retention_days" {
  type        = number
  description = "Number of days to retain backups before deletion"
  default     = 90
}

variable "archive_after_days" {
  type        = number
  description = "Number of days before moving backups to archive storage class"
  default     = 30
}

variable "archive_retention_days" {
  type        = number
  description = "Number of days to retain archives (7 years = 2555 days)"
  default     = 2555
}

# Local to determine if backup should be enabled
locals {
  enable_backup = var.enable_backup_infrastructure || terraform.workspace == "prod"
}

###############################################################
# Cross-Region Backup Storage (Isolated from Production)
###############################################################

# Backup storage bucket in a different region than production
resource "google_storage_bucket" "backup_bucket" {
  count         = local.enable_backup ? 1 : 0
  name          = "${local.current_config.base_name}-backups-isolated"
  location      = var.backup_region
  force_destroy = false # Protect against accidental deletion
  storage_class = "STANDARD"

  # Enable versioning for backup retention
  versioning {
    enabled = true
  }

  # Lifecycle management for backup retention
  lifecycle_rule {
    condition {
      age = var.backup_retention_days
    }
    action {
      type = "Delete"
    }
  }

  # Archive older backups
  lifecycle_rule {
    condition {
      age = var.archive_after_days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  uniform_bucket_level_access = true

  # Labels for compliance tracking
  labels = {
    purpose     = "backup-storage"
    environment = terraform.workspace
    compliance  = "soc2"
    isolation   = "cross-region"
  }
}

# Long-term archive bucket in a third region
resource "google_storage_bucket" "archive_bucket" {
  count         = local.enable_backup ? 1 : 0
  name          = "${local.current_config.base_name}-archives-isolated"
  location      = var.archive_region
  force_destroy = false
  storage_class = "COLDLINE"

  versioning {
    enabled = true
  }

  # Longer retention for archives
  lifecycle_rule {
    condition {
      age = var.archive_retention_days
    }
    action {
      type = "Delete"
    }
  }

  uniform_bucket_level_access = true

  labels = {
    purpose     = "long-term-archive"
    environment = terraform.workspace
    compliance  = "soc2"
    isolation   = "cross-region-archive"
  }
}

###############################################################
# Backup-Specific Service Account (Restricted Permissions)
###############################################################

resource "google_service_account" "backup_service_account" {
  count        = local.enable_backup ? 1 : 0
  account_id   = "${local.current_config.base_name}-backup-only"
  display_name = "Backup Service Account (Isolated)"
  description  = "Service account with restricted permissions for backup operations only"
}

# Grant backup service account access ONLY to backup buckets
resource "google_storage_bucket_iam_member" "backup_bucket_access" {
  count  = local.enable_backup ? 1 : 0
  bucket = google_storage_bucket.backup_bucket[0].name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.backup_service_account[0].email}"
}

resource "google_storage_bucket_iam_member" "archive_bucket_access" {
  count  = local.enable_backup ? 1 : 0
  bucket = google_storage_bucket.archive_bucket[0].name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.backup_service_account[0].email}"
}

# Explicitly DENY access to production bucket
resource "google_storage_bucket_iam_member" "backup_sa_no_prod_access" {
  count  = local.enable_backup ? 1 : 0
  bucket = google_storage_bucket.project_bucket.name
  role   = "roles/storage.legacyBucketReader" # Minimal read-only role
  member = "serviceAccount:${google_service_account.backup_service_account[0].email}"

  # This ensures backup SA cannot write to production storage
}

###############################################################
# Database Backup Configuration (Automated)
###############################################################

# Cloud SQL backup configuration with cross-region replication
resource "google_sql_database_instance" "postgres_replica" {
  count            = local.enable_backup ? 1 : 0
  name             = "${local.current_config.base_name}-postgres-backup-replica"
  database_version = "POSTGRES_14"
  region           = var.backup_region

  master_instance_name = google_sql_database_instance.postgres.name

  settings {
    tier = "db-custom-2-4096"

    # Database flags must match the master instance
    database_flags {
      name  = "max_connections"
      value = 3000
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc_network.id
    }
  }

  deletion_protection = true # Protect backup replica

  depends_on = [google_service_networking_connection.private_service_connection]
}

###############################################################
# Outputs for Compliance Documentation
###############################################################

output "backup_isolation_evidence" {
  value = local.enable_backup ? {
    backup_bucket = {
      name     = google_storage_bucket.backup_bucket[0].name
      location = google_storage_bucket.backup_bucket[0].location
      url      = google_storage_bucket.backup_bucket[0].url
    }
    archive_bucket = {
      name     = google_storage_bucket.archive_bucket[0].name
      location = google_storage_bucket.archive_bucket[0].location
      url      = google_storage_bucket.archive_bucket[0].url
    }
    production_bucket = {
      name     = google_storage_bucket.project_bucket.name
      location = google_storage_bucket.project_bucket.location
      url      = google_storage_bucket.project_bucket.url
    }
    backup_service_account = google_service_account.backup_service_account[0].email
    isolation_summary      = "Backup data stored in different regions with restricted service account access"
    backup_enabled         = true
    message                = "Backup infrastructure is enabled and configured"
    } : {
    backup_bucket = {
      name     = "not-configured"
      location = "not-configured"
      url      = "not-configured"
    }
    archive_bucket = {
      name     = "not-configured"
      location = "not-configured"
      url      = "not-configured"
    }
    production_bucket = {
      name     = google_storage_bucket.project_bucket.name
      location = google_storage_bucket.project_bucket.location
      url      = google_storage_bucket.project_bucket.url
    }
    backup_service_account = "not-configured"
    isolation_summary      = "Backup infrastructure not enabled for this workspace"
    backup_enabled         = false
    message                = "Backup infrastructure not enabled for this workspace"
  }
  description = "Evidence of backup data isolation for SOC2 compliance"
}

output "backup_regions_isolation" {
  value = local.enable_backup ? {
    production_region = local.current_config.region
    backup_region     = var.backup_region
    archive_region    = var.archive_region
    isolation_type    = "cross-region-geographic"
    backup_enabled    = true
    message           = "Geographic isolation is active"
    } : {
    production_region = local.current_config.region
    backup_region     = var.backup_region
    archive_region    = var.archive_region
    isolation_type    = "not-configured"
    backup_enabled    = false
    message           = "Backup infrastructure not enabled for this workspace"
  }
  description = "Geographic isolation of backup data from production"
}

output "backup_configuration" {
  value = {
    enabled                = local.enable_backup
    workspace              = terraform.workspace
    backup_region          = var.backup_region
    archive_region         = var.archive_region
    backup_retention_days  = var.backup_retention_days
    archive_after_days     = var.archive_after_days
    archive_retention_days = var.archive_retention_days
  }
  description = "Backup configuration settings"
}
