###############################################################
# Anti-malware & Intrusion Detection (Compliance Control)
###############################################################

# Enable Security Command Center API (free tier)
resource "google_project_service" "security_center" {
  project            = local.current_config.project
  service            = "securitycenter.googleapis.com"
  disable_on_destroy = false
}

# Enable container vulnerability scanning (free)
resource "google_project_service" "container_analysis" {
  project            = local.current_config.project
  service            = "containeranalysis.googleapis.com"
  disable_on_destroy = false
}


# Output for evidence collection
output "security_command_center_url" {
  value       = "https://console.cloud.google.com/security/command-center/overview?project=${local.current_config.project}"
  description = "URL to Security Command Center dashboard for compliance evidence"
}
