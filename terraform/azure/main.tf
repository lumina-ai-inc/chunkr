terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

variable "base_name" {
  default = "chunkr"
}

variable "location" {
  default = "eastus2"
}

variable "postgres_username" {
  type        = string
  description = "The username for the PostgreSQL database"
}

variable "postgres_password" {
  type        = string
  description = "The password for the PostgreSQL database"
  sensitive   = true
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
  default = 6
}

variable "general_vm_size" {
  default = "Standard_F8s_v2" # Similar to GCP c2d-highcpu-4
}

variable "gpu_vm_count" {
  default = 1
}

variable "gpu_min_vm_count" {
  default = 1
}

variable "gpu_max_vm_count" {
  default = 6
}

variable "gpu_vm_size" {
  default = "Standard_NC8as_T4_v3" # Similar to GCP a2-highgpu-1g
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.base_name}-rg"
  location = var.location
}

###############################################################
# Virtual Network
###############################################################
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.base_name}-vnet"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  address_space       = ["10.3.0.0/16"]
}

resource "azurerm_subnet" "aks_subnet" {
  name                 = "${var.base_name}-aks-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.3.1.0/24"]
}

resource "azurerm_subnet" "services_subnet" {
  name                 = "${var.base_name}-services-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.3.2.0/24"]

  delegation {
    name = "fs"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
      ]
    }
  }
}

###############################################################
# Redis Cache
###############################################################
resource "azurerm_redis_cache" "cache" {
  name                = "${var.base_name}-redis"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  capacity            = 1
  family              = "C"
  sku_name            = "Basic"
}

###############################################################
# Storage Account 
###############################################################
resource "random_string" "storage_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "azurerm_storage_account" "storage" {
  name                     = "${replace(lower(var.base_name), "-", "")}${random_string.storage_suffix.result}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  blob_properties {
    cors_rule {
      allowed_headers    = ["*"]
      allowed_methods    = ["GET", "HEAD", "POST"]
      allowed_origins    = ["*"]
      exposed_headers    = ["*"]
      max_age_in_seconds = 3600
    }
  }
}

resource "azurerm_storage_container" "container" {
  name                  = "${var.base_name}-container"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

###############################################################
# AKS Cluster
###############################################################
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.base_name}-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = var.base_name

  default_node_pool {
    name                = "system"
    node_count          = 1
    vm_size             = "Standard_D2s_v3"
    vnet_subnet_id      = azurerm_subnet.aks_subnet.id
    enable_auto_scaling = false
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "general" {
  name                  = "general"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.general_vm_size
  node_count            = var.general_vm_count
  enable_auto_scaling   = true
  min_count             = var.general_min_vm_count
  max_count             = var.general_max_vm_count
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id

  node_labels = {
    "purpose" = "general-compute"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_vm_count
  enable_auto_scaling   = true
  min_count             = var.gpu_min_vm_count
  max_count             = var.gpu_max_vm_count
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id

  node_labels = {
    "purpose" = "gpu-compute"
  }

  node_taints = [
    "nvidia.com/gpu=present:NoSchedule"
  ]
}

###############################################################
# PostgreSQL
###############################################################
resource "azurerm_postgresql_flexible_server" "postgres" {
  name                          = "${var.base_name}-postgres"
  resource_group_name           = azurerm_resource_group.rg.name
  location                      = azurerm_resource_group.rg.location
  version                       = "14"
  delegated_subnet_id           = azurerm_subnet.services_subnet.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres.id
  administrator_login           = var.postgres_username
  administrator_password        = var.postgres_password
  zone                          = "1"
  public_network_access_enabled = false

  storage_mb = 32768

  sku_name = "B_Standard_B2s"
}

resource "azurerm_private_dns_zone" "postgres" {
  name                = "${var.base_name}-postgres.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${var.base_name}-postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  resource_group_name   = azurerm_resource_group.rg.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
}

resource "azurerm_postgresql_flexible_server_database" "chunkr" {
  name      = var.chunkr_db
  server_id = azurerm_postgresql_flexible_server.postgres.id

  depends_on = [azurerm_postgresql_flexible_server.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "keycloak" {
  name      = var.keycloak_db
  server_id = azurerm_postgresql_flexible_server.postgres.id

  depends_on = [azurerm_postgresql_flexible_server.postgres]
}

###############################################################
# Outputs
###############################################################
output "redis_connection_string" {
  value     = azurerm_redis_cache.cache.primary_connection_string
  sensitive = true
}

output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}

output "storage_account_key" {
  value     = azurerm_storage_account.storage.primary_access_key
  sensitive = true
}

output "postgres_server_name" {
  value = azurerm_postgresql_flexible_server.postgres.name
}

output "postgres_connection_string" {
  value     = "postgresql://${var.postgres_username}:${var.postgres_password}@${azurerm_postgresql_flexible_server.postgres.fqdn}:5432/${var.chunkr_db}"
  sensitive = true
}

output "keycloak_connection_string" {
  value     = "postgresql://${azurerm_postgresql_flexible_server.postgres.fqdn}:5432/${var.keycloak_db}"
  sensitive = true
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "aks_resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "aks_connection_command" {
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.rg.name} --name ${azurerm_kubernetes_cluster.aks.name}"
  description = "Command to configure kubectl to connect to the AKS cluster"
}