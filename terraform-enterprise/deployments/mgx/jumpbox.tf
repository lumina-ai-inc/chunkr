###############################################################
# Jumpbox VM for AKS Access
###############################################################

# Create a subnet for the jumpbox VM
resource "azurerm_subnet" "jumpbox_subnet" {
  name                 = "${var.base_name}-jumpbox-subnet"
  resource_group_name  = data.azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.3.30.0/24"]
}

# Network Security Group for jumpbox
resource "azurerm_network_security_group" "jumpbox_nsg" {
  name                = "${var.base_name}-jumpbox-nsg"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Network interface for jumpbox VM
resource "azurerm_network_interface" "jumpbox_nic" {
  name                = "${var.base_name}-jumpbox-nic"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.jumpbox_subnet.id
    private_ip_address_allocation = "Dynamic"
  }
}

# Associate NSG with network interface
resource "azurerm_network_interface_security_group_association" "jumpbox_nsg_association" {
  network_interface_id      = azurerm_network_interface.jumpbox_nic.id
  network_security_group_id = azurerm_network_security_group.jumpbox_nsg.id
}

# Jumpbox VM
resource "azurerm_linux_virtual_machine" "jumpbox" {
  name                = "${var.base_name}-jumpbox"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  size                = "Standard_B16ms"
  admin_username      = "adminuser"

  admin_ssh_key {
    username   = "adminuser"
    public_key = file("~/.ssh/id_rsa.pub") # Make sure this path is correct or use a variable
  }

  network_interface_ids = [
    azurerm_network_interface.jumpbox_nic.id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  custom_data = base64encode(<<-EOF
    #!/bin/bash
    # Install Azure CLI
    curl -sL https://aka.ms/InstallAzureCLI | bash
    
    # Install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    
    # Install Helm
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  EOF
  )
}

# Output the Jumpbox private IP
output "jumpbox_private_ip" {
  value = azurerm_linux_virtual_machine.jumpbox.private_ip_address
}

# Instructions for connecting to the Jumpbox
output "jumpbox_connection_instructions" {
  value = "Connect to the jumpbox via Azure Bastion. Once connected, use 'az login' and 'az aks get-credentials --resource-group ${data.azurerm_resource_group.rg.name} --name ${azurerm_kubernetes_cluster.aks.name}' to access the AKS cluster."
}

# Output the complete command to connect to the jumpbox VM using Azure Bastion with Entra ID
# Output the complete command to connect to the jumpbox VM using Azure Bastion with SSH key
output "jumpbox_bastion_ssh_connect_command" {
  description = "Ready-to-use Azure CLI command to connect to the jumpbox VM using Azure Bastion with SSH key"
  value       = "az network bastion ssh --name \"${azurerm_bastion_host.bastion.name}\" --resource-group \"${data.azurerm_resource_group.rg.name}\" --target-resource-id \"${azurerm_linux_virtual_machine.jumpbox.id}\" --auth-type \"ssh-key\" --username adminuser --ssh-key ~/.ssh/id_rsa"
}
