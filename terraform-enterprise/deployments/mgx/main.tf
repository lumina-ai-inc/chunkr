# Configuration has been split into separate files:
# - provider.tf: Terraform providers configuration
# - variables.tf: Variable declarations
# - networking.tf: Virtual network, subnets, and NSGs
# - firewall.tf: Azure Firewall configuration
# - aks.tf: AKS cluster and node pools
# - postgres.tf: PostgreSQL resources
# - acr.tf: Azure Container Registry
# - security.tf: Microsoft Defender, Sentinel, and Log Analytics
# - outputs.tf: Output values
# - jumpbox.tf: Jumpbox VM

# Reference existing Key Vault
data "azurerm_key_vault" "chunkr_vault" {
  name                = "mgx-chunkr-keyvault"
  resource_group_name = var.resource_group_name
}
