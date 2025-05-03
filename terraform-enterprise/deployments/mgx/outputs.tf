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
  value     = var.create_postgres ? azurerm_postgresql_flexible_server.postgres[0].administrator_password : null
  sensitive = true
}

output "postgres_connection_string" {
  value     = var.create_postgres ? "postgresql://${var.postgres_username}:${azurerm_postgresql_flexible_server.postgres[0].administrator_password}@${azurerm_postgresql_flexible_server.postgres[0].fqdn}:5432/${var.chunkr_db}" : null
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
  value = data.azurerm_resource_group.rg.name
}

output "aks_connection_command" {
  value       = "az aks get-credentials --resource-group ${data.azurerm_resource_group.rg.name} --name ${azurerm_kubernetes_cluster.aks.name}"
  description = "Command to configure kubectl to connect to the AKS cluster"
}

output "key_vault_secrets_provider_client_id" {
  description = "Client ID of the Key Vault Secrets Provider managed identity"
  value       = azurerm_kubernetes_cluster.aks.key_vault_secrets_provider[0].secret_identity[0].client_id
  sensitive   = true
}

output "azure_tenant_id" {
  description = "Azure tenant ID"
  value       = data.azurerm_client_config.current.tenant_id
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = data.azurerm_key_vault.chunkr_vault.name
}
