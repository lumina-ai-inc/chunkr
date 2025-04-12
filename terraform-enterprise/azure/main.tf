terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  backend "s3" {}
}

variable "base_name" {
  type        = string
  description = "The base name for the resources"
}

variable "location" {
  default = "eastus2"
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

variable "general_vm_size" {
  default = "Standard_F8s_v2"
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

variable "gpu_vm_size" {
  default = "Standard_NC8as_T4_v3"
}

variable "create_postgres" {
  description = "Whether to create PostgreSQL resources"
  type        = bool
  default     = false
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
  address_prefixes     = ["10.3.0.0/20"]
}

resource "azurerm_subnet" "services_subnet" {
  name                 = "${var.base_name}-services-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.3.16.0/22"]

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
# Network Security Group
###############################################################
resource "azurerm_network_security_group" "services_nsg" {
  name                = "${var.base_name}-services-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "allow-http"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "allow-https"
    priority                   = 101
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "allow-icmp"
    priority                   = 102
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Icmp"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Associate the NSG with the services subnet
resource "azurerm_subnet_network_security_group_association" "services_nsg_association" {
  subnet_id                 = azurerm_subnet.services_subnet.id
  network_security_group_id = azurerm_network_security_group.services_nsg.id
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
    name           = "system"
    node_count     = 1
    vm_size        = "Standard_D2s_v3"
    vnet_subnet_id = azurerm_subnet.aks_subnet.id
    upgrade_settings {
      drain_timeout_in_minutes      = 0
      max_surge                     = "10%"
      node_soak_duration_in_minutes = 0
    }
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
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id

  node_labels = {
    "purpose" = "general-compute"
  }

  max_pods = 250
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_vm_count
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id

  node_labels = {
    "purpose" = "gpu-compute"
  }

  node_taints = [
    "nvidia.com/gpu=present:NoSchedule"
  ]

  max_pods = 250
}

###############################################################
# PostgreSQL
###############################################################
resource "azurerm_postgresql_flexible_server" "postgres" {
  count                         = var.create_postgres ? 1 : 0
  name                          = "${var.base_name}-postgres"
  resource_group_name           = azurerm_resource_group.rg.name
  location                      = azurerm_resource_group.rg.location
  version                       = "14"
  delegated_subnet_id           = azurerm_subnet.services_subnet.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres[0].id
  administrator_login           = var.postgres_username
  administrator_password        = var.postgres_password
  zone                          = "1"
  public_network_access_enabled = false

  storage_mb = 32768

  sku_name = "B_Standard_B2s"
}

resource "azurerm_private_dns_zone" "postgres" {
  count               = var.create_postgres ? 1 : 0
  name                = "${var.base_name}-postgres.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  count                 = var.create_postgres ? 1 : 0
  name                  = "${var.base_name}-postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres[0].name
  resource_group_name   = azurerm_resource_group.rg.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
}

resource "azurerm_postgresql_flexible_server_database" "chunkr" {
  count     = var.create_postgres ? 1 : 0
  name      = var.chunkr_db
  server_id = azurerm_postgresql_flexible_server.postgres[0].id

  depends_on = [azurerm_postgresql_flexible_server.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "keycloak" {
  count     = var.create_postgres ? 1 : 0
  name      = var.keycloak_db
  server_id = azurerm_postgresql_flexible_server.postgres[0].id

  depends_on = [azurerm_postgresql_flexible_server.postgres]
}

resource "azurerm_postgresql_flexible_server_configuration" "uuid_ossp" {
  count     = var.create_postgres ? 1 : 0
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.postgres[0].id
  value     = "UUID-OSSP"
}

###############################################################
# Outputs
###############################################################
output "postgres_server_name" {
  value = var.create_postgres ? azurerm_postgresql_flexible_server.postgres[0].name : null
}

output "postgres_server_username" {
  value = var.create_postgres ? var.postgres_username : null
}

output "postgres_server_password" {
  value     = var.create_postgres ? var.postgres_password : null
  sensitive = true
}

output "postgres_connection_string" {
  value     = var.create_postgres ? "postgresql://${var.postgres_username}:${var.postgres_password}@${azurerm_postgresql_flexible_server.postgres[0].fqdn}:5432/${var.chunkr_db}" : null
  sensitive = true
}

output "keycloak_connection_string" {
  value     = var.create_postgres ? "jdbc:postgresql://${azurerm_postgresql_flexible_server.postgres[0].fqdn}:5432/${var.keycloak_db}" : null
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
