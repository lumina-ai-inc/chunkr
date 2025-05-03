variable "base_name" {
  type        = string
  description = "The base name for the resources"
}

variable "location" {
  type        = string
  default     = "eastus2"
  description = "The location for the resources"
}

variable "postgres_username" {
  type        = string
  description = "The username for the PostgreSQL database"
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

variable "resource_group_name" {
  type        = string
  description = "The name of the existing resource group to use"
}
