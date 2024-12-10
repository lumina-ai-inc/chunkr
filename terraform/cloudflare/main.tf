terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

###############################################################
# Cloudflare Tunnel
###############################################################

resource "random_id" "tunnel_secret" {
  byte_length = 35
}

resource "cloudflare_tunnel" "chunkr_tunnel" {
  account_id = var.cloudflare_account_id
  name       = "chunkr-tunnel"
  secret     = random_id.tunnel_secret.b64_std
}

###############################################################
# Outputs
###############################################################

output "tunnel_id" {
  description = "ID of the created Cloudflare Tunnel"
  value       = cloudflare_tunnel.chunkr_tunnel.id
}

output "tunnel_name" {
  description = "Name of the created Cloudflare Tunnel"
  value       = cloudflare_tunnel.chunkr_tunnel.name
}

output "tunnel_cname" {
  description = "CNAME target for the tunnel"
  value       = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
}
