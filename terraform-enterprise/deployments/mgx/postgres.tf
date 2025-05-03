###############################################################
# PostgreSQL
###############################################################
resource "random_password" "postgres_password" {
  length  = 16
  special = false
}

resource "azurerm_postgresql_flexible_server" "postgres" {
  count                         = var.create_postgres ? 1 : 0
  name                          = "${var.base_name}-postgres"
  resource_group_name           = data.azurerm_resource_group.rg.name
  location                      = data.azurerm_resource_group.rg.location
  version                       = "14"
  delegated_subnet_id           = azurerm_subnet.services_subnet.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres[0].id
  administrator_login           = var.postgres_username
  administrator_password        = random_password.postgres_password.result
  zone                          = "1"
  public_network_access_enabled = false

  storage_mb = 32768

  sku_name = "B_Standard_B2s"

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = true
  }
}

resource "azurerm_private_dns_zone" "postgres" {
  count               = var.create_postgres ? 1 : 0
  name                = "${var.base_name}-postgres.private.postgres.database.azure.com"
  resource_group_name = data.azurerm_resource_group.rg.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  count                 = var.create_postgres ? 1 : 0
  name                  = "${var.base_name}-postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres[0].name
  resource_group_name   = data.azurerm_resource_group.rg.name
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
