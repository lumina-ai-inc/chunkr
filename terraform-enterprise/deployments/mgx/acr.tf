###############################################################
# Azure Container Registry
###############################################################

resource "azurerm_container_registry" "acr" {
  name                = "${replace(var.base_name, "-", "")}acr"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  sku                 = "Premium" # Required for private endpoints
  admin_enabled       = false

  identity {
    type = "SystemAssigned"
  }
}

# Update the private endpoint to use the endpoint subnet
resource "azurerm_private_endpoint" "acr_endpoint" {
  name                = "${var.base_name}-acr-endpoint"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.endpoint_subnet.id

  private_service_connection {
    name                           = "${var.base_name}-acr-connection"
    private_connection_resource_id = azurerm_container_registry.acr.id
    is_manual_connection           = false
    subresource_names              = ["registry"]
  }
}

# Enable scanning on push
resource "azurerm_container_registry_scope_map" "acr_scan_scope" {
  name                    = "${var.base_name}-scan-scope"
  container_registry_name = azurerm_container_registry.acr.name
  resource_group_name     = data.azurerm_resource_group.rg.name
  actions                 = ["repositories/*/metadata/read"]
}

# Grant AKS cluster access to pull images from ACR
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}
