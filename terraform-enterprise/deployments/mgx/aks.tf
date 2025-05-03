###############################################################
# AKS Cluster
###############################################################
resource "azurerm_kubernetes_cluster" "aks" {
  name                      = "${var.base_name}-aks"
  location                  = data.azurerm_resource_group.rg.location
  resource_group_name       = data.azurerm_resource_group.rg.name
  dns_prefix                = var.base_name
  private_cluster_enabled   = true
  automatic_channel_upgrade = "stable"

  default_node_pool {
    name            = "system"
    node_count      = 1
    vm_size         = "Standard_D2s_v3"
    vnet_subnet_id  = azurerm_subnet.aks_subnet.id
    os_disk_size_gb = 128
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
    outbound_type  = "loadBalancer"
    # outbound_type  = "userDefinedRouting"
  }

  microsoft_defender {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.workspace.id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "general" {
  name                  = "general"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.general_vm_size
  node_count            = var.general_vm_count
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id
  os_disk_size_gb       = 128

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
  os_disk_size_gb       = 256

  node_labels = {
    "purpose" = "gpu-compute"
  }

  node_taints = [
    "nvidia.com/gpu:NoSchedule"
  ]

  max_pods = 250
}

resource "azurerm_key_vault_access_policy" "aks_policy" {
  key_vault_id = data.azurerm_key_vault.chunkr_vault.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_kubernetes_cluster.aks.identity[0].principal_id

  secret_permissions = [
    "Get",
    "List"
  ]
}

resource "azurerm_key_vault_access_policy" "kubelet_policy" {
  key_vault_id = data.azurerm_key_vault.chunkr_vault.id
  tenant_id    = data.azurerm_client_config.current.tenant_id

  # This is the kubelet identity's object ID
  object_id = "96310e76-febe-43a8-849b-299d1597adf0"

  secret_permissions = [
    "Get",
    "List"
  ]
}

data "azurerm_client_config" "current" {}

output "aks_identity_principal_id" {
  value = azurerm_kubernetes_cluster.aks.identity[0].principal_id
}
