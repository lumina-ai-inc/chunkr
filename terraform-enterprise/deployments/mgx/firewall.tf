# Create a subnet for Azure Firewall
resource "azurerm_subnet" "firewall_subnet" {
  name                 = "AzureFirewallSubnet"
  resource_group_name  = data.azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.3.24.0/24"]
}

# Create a public IP for Azure Firewall
resource "azurerm_public_ip" "firewall_ip" {
  name                = "${var.base_name}-firewall-ip"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

# Deploy Azure Firewall
resource "azurerm_firewall" "firewall" {
  name                = "${var.base_name}-firewall"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  sku_name            = "AZFW_VNet"
  sku_tier            = "Premium"
  firewall_policy_id  = azurerm_firewall_policy.policy.id

  ip_configuration {
    name                 = "configuration"
    subnet_id            = azurerm_subnet.firewall_subnet.id
    public_ip_address_id = azurerm_public_ip.firewall_ip.id
  }
}

# Reorder resources to ensure the policy is created before the firewall
resource "azurerm_firewall_policy" "policy" {
  name                = "${var.base_name}-firewall-policy"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location

  sku                      = "Premium"
  threat_intelligence_mode = "Deny"

  intrusion_detection {
    mode = "Alert"
  }
}

# Create a route table
resource "azurerm_route_table" "aks_route_table" {
  name                = "${var.base_name}-aks-route-table"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
}

# Use a known IP address for the route or handle with depends_on
resource "azurerm_route" "firewall_route" {
  name                   = "firewall-route"
  resource_group_name    = data.azurerm_resource_group.rg.name
  route_table_name       = azurerm_route_table.aks_route_table.name
  address_prefix         = "0.0.0.0/0"
  next_hop_type          = "VirtualAppliance"
  next_hop_in_ip_address = "10.3.24.4" # Set a static IP within the firewall subnet
  depends_on             = [azurerm_firewall.firewall]
}

# Associate route table with AKS subnet
resource "azurerm_subnet_route_table_association" "aks_subnet_rt_association" {
  subnet_id      = azurerm_subnet.aks_subnet.id
  route_table_id = azurerm_route_table.aks_route_table.id
}

# Associate route table with services subnet
resource "azurerm_subnet_route_table_association" "services_subnet_rt_association" {
  subnet_id      = azurerm_subnet.services_subnet.id
  route_table_id = azurerm_route_table.aks_route_table.id
}

resource "azurerm_firewall_policy_rule_collection_group" "network_rules" {
  name               = "${var.base_name}-network-rules"
  firewall_policy_id = azurerm_firewall_policy.policy.id
  priority           = 200

  network_rule_collection {
    name     = "aks-required-services"
    priority = 100
    action   = "Allow"

    rule {
      name                  = "allow-dns"
      protocols             = ["UDP", "TCP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["*"]
      destination_ports     = ["53"]
    }

    rule {
      name             = "allow-azure-services"
      protocols        = ["TCP"]
      source_addresses = ["10.3.0.0/16"]
      destination_addresses = [
        "AzureContainerRegistry",
        "MicrosoftContainerRegistry",
        "AzureActiveDirectory"
      ]
      destination_ports = ["443"]
    }

    rule {
      name                  = "allow-ntp"
      protocols             = ["UDP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["*"]
      destination_ports     = ["123"]
    }
  }

  network_rule_collection {
    name     = "aks-additional-services"
    priority = 110
    action   = "Allow"

    rule {
      name                  = "allow-aks-services"
      protocols             = ["TCP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["AzureKubernetesService"]
      destination_ports     = ["443"]
    }

    rule {
      name                  = "allow-aks-health"
      protocols             = ["TCP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["*"]
      destination_ports     = ["9000"]
    }
  }

  network_rule_collection {
    name     = "cloudflared-connections"
    priority = 120
    action   = "Allow"

    rule {
      name                  = "allow-cloudflared-tcp"
      protocols             = ["TCP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["198.41.192.0/19", "198.41.200.0/22"]
      destination_ports     = ["443", "7844"]
    }

    rule {
      name                  = "allow-cloudflared-udp"
      protocols             = ["UDP"]
      source_addresses      = ["10.3.0.0/16"]
      destination_addresses = ["198.41.192.0/19", "198.41.200.0/22"]
      destination_ports     = ["443", "7844"]
    }

    rule {
      name             = "allow-cloudflared-quic-region1"
      protocols        = ["UDP"]
      source_addresses = ["10.3.0.0/16"]
      destination_addresses = [
        "198.41.192.167", "198.41.192.67", "198.41.192.57", "198.41.192.107",
        "198.41.192.27", "198.41.192.7", "198.41.192.227", "198.41.192.47",
        "198.41.192.37", "198.41.192.77"
      ]
      destination_ports = ["443", "7844"]
    }

    rule {
      name             = "allow-cloudflared-quic-region2"
      protocols        = ["UDP"]
      source_addresses = ["10.3.0.0/16"]
      destination_addresses = [
        "198.41.200.13", "198.41.200.193", "198.41.200.33", "198.41.200.233",
        "198.41.200.53", "198.41.200.63", "198.41.200.113", "198.41.200.73",
        "198.41.200.43", "198.41.200.23"
      ]
      destination_ports = ["443", "7844"]
    }
  }
}

# Add application rules for AKS
resource "azurerm_firewall_policy_rule_collection_group" "app_rules" {
  name               = "${var.base_name}-app-rules"
  firewall_policy_id = azurerm_firewall_policy.policy.id
  priority           = 300

  application_rule_collection {
    name     = "aks-fqdns"
    priority = 100
    action   = "Allow"

    rule {
      name = "allow-aks-required-fqdns"
      protocols {
        type = "Https"
        port = 443
      }
      protocols {
        type = "Http"
        port = 80
      }
      source_addresses = ["10.3.0.0/16"]
      destination_fqdns = [
        "*.hcp.${data.azurerm_resource_group.rg.location}.azmk8s.io",
        "mcr.microsoft.com",
        "*.data.mcr.microsoft.com",
        "management.azure.com",
        "login.microsoftonline.com",
        "packages.microsoft.com",
        "acs-mirror.azureedge.net",
        "*.ubuntu.com",
        "security.ubuntu.com",
        "azure.archive.ubuntu.com",
        "changelogs.ubuntu.com",
        "dc.services.visualstudio.com",
        "*.ods.opinsights.azure.com",
        "*.oms.opinsights.azure.com",
        "*.monitoring.azure.com",
        "github.com",
        "gcr.io",
        "*.azurecr.io",
        "k8s.gcr.io",
        "registry-1.docker.io",
        "auth.docker.io",
        "production.cloudflare.docker.com",
        "*.docker.io",
        "quay.io",
        "*.quay.io",
        "*.blob.core.windows.net",
        "*.cloudflareclient.com",
        "*.cloudflare.com",
        "*.trycloudflare.com",
        "region1.v2.argotunnel.com",
        "region2.v2.argotunnel.com",
        "*.cftunnel.com",
        "h2.cftunnel.com",
        "quic.cftunnel.com",
        "api.cloudflare.com",
        "update.argotunnel.com",
        "pqtunnels.cloudflareresearch.com",
        "*.chunkr.ai",
        "*.api.cognitive.microsoft.com",
        "*.openai.com"
      ]
    }
  }

  application_rule_collection {
    name     = "os-updates"
    priority = 200
    action   = "Allow"

    rule {
      name = "allow-os-updates"
      protocols {
        type = "Https"
        port = 443
      }
      protocols {
        type = "Http"
        port = 80
      }
      source_addresses = ["10.3.0.0/16"]
      destination_fqdns = [
        "download.opensuse.org",
        "*.download.opensuse.org",
        "api.snapcraft.io",
        "motd.ubuntu.com"
      ]
    }
  }
}
