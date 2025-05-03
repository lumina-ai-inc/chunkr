###############################################################
# Microsoft Defender for Cloud
############################################################### 

# Create Log Analytics workspace for Defender
resource "azurerm_log_analytics_workspace" "workspace" {
  name                = "${var.base_name}-workspace"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# Enable Microsoft Defender for Cloud
# resource "azurerm_security_center_subscription_pricing" "defender_for_containers" {
#   tier          = "Standard"
#   resource_type = "KubernetesService"
# }

# Enable Microsoft Defender for Container Registries
# resource "azurerm_security_center_subscription_pricing" "defender_for_acr" {
#   tier          = "Standard"
#   resource_type = "ContainerRegistry"
# }

###############################################################
# Azure Sentinel
###############################################################

# Configure Defender to send alerts to Log Analytics
# resource "azurerm_security_center_workspace" "defender_workspace" {
#   scope        = "/subscriptions/${data.azurerm_subscription.current.subscription_id}"
#   workspace_id = azurerm_log_analytics_workspace.workspace.id
# }

# Configure diagnostic settings for AKS to send logs to workspace
resource "azurerm_monitor_diagnostic_setting" "aks_diagnostics" {
  name                       = "${var.base_name}-aks-diagnostics"
  target_resource_id         = azurerm_kubernetes_cluster.aks.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.workspace.id

  enabled_log {
    category = "kube-apiserver"
  }

  enabled_log {
    category = "kube-audit"
  }

  enabled_log {
    category = "kube-controller-manager"
  }

  enabled_log {
    category = "kube-scheduler"
  }

  enabled_log {
    category = "cluster-autoscaler"
  }

  metric {
    category = "AllMetrics"
  }
}
