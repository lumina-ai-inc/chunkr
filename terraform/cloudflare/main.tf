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
# DNS Records - Optional
###############################################################

resource "cloudflare_record" "root" {
  count   = var.cloudflare_zone_id != "" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "@"
  content = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_record" "www" {
  count   = var.cloudflare_zone_id != "" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "www"
  content = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_record" "api" {
  count   = var.cloudflare_zone_id != "" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "api"
  content = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_record" "auth" {
  count   = var.cloudflare_zone_id != "" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "auth"
  content = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_record" "rrq" {
  count   = var.cloudflare_zone_id != "" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "rrq"
  content = "${cloudflare_tunnel.chunkr_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
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

output "domain_configured" {
  description = "Whether domain records were configured"
  value       = var.cloudflare_zone_id != ""
}

output "dns_records" {
  description = "Map of configured DNS records (only if domain is configured)"
  value = var.cloudflare_zone_id != "" ? {
    root = "https://${var.domain}"
    www  = "https://www.${var.domain}"
    api  = "https://api.${var.domain}"
    auth = "https://auth.${var.domain}"
    rrq  = "https://rrq.${var.domain}"
  } : null
}
