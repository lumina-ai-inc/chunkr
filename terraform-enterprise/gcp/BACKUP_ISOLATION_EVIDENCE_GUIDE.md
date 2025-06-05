# SOC2 Recovery Data Isolation - Step-by-Step Evidence Guide

This guide provides detailed instructions on how to demonstrate that your recovery data is isolated from your production environment, satisfying SOC2 compliance requirements.

## 🎯 Control Objective

**Isolate the recovery data from the production environment to prevent accidental overwriting or corruption of backups.**

## 📋 Prerequisites

Before collecting evidence, ensure you have:
- [ ] GCP CLI (`gcloud`) installed and authenticated
- [ ] Terraform access to your infrastructure
- [ ] Access to the `staging` or `dev` workspace (NOT production)
- [ ] Appropriate permissions to view storage buckets and IAM policies

## 🔧 Backup Infrastructure Configuration

### Default Behavior
- **Production workspace (`prod`)**: Backup infrastructure is **automatically enabled**
- **Other workspaces**: Backup infrastructure is **disabled** by default (cost optimization)

### Configuration Variables

The backup isolation infrastructure is configurable via Terraform variables:

```hcl
# Enable backup infrastructure for non-prod environments
enable_backup_infrastructure = true

# Configure backup regions (should be different from production)
backup_region = "us-central1"    # Default
archive_region = "us-east1"      # Default

# Configure retention policies
backup_retention_days = 90       # Default: 90 days
archive_after_days = 30          # Default: 30 days  
archive_retention_days = 2555    # Default: 7 years
```

### Enabling Backup Infrastructure for Testing

To enable backup infrastructure for testing in staging/dev:

```bash
# Option 1: Set via terraform.tfvars
echo 'enable_backup_infrastructure = true' >> terraform.tfvars

# Option 2: Set via command line
./tf-workspace.sh staging apply -var="enable_backup_infrastructure=true"

# Option 3: Set custom regions
./tf-workspace.sh staging apply \
  -var="enable_backup_infrastructure=true" \
  -var="backup_region=us-west2" \
  -var="archive_region=us-east4"
```

## 🚀 Step-by-Step Evidence Collection

### Step 1: Deploy Backup Isolation Infrastructure (Optional)

For **production environments**, backup infrastructure is automatically deployed.

For **testing/demonstration** in staging, optionally enable backup infrastructure:

```bash
# Navigate to the terraform directory
cd terraform-enterprise/gcp

# Option A: Deploy without backup infrastructure (default for staging)
./tf-workspace.sh staging plan
./tf-workspace.sh staging apply

# Option B: Deploy WITH backup infrastructure for testing
./tf-workspace.sh staging apply -var="enable_backup_infrastructure=true"
```

### Step 2: Generate Evidence Package

Run the automated evidence collection script:

```bash
# Make the script executable (if not already done)
chmod +x generate-backup-evidence.sh

# Generate evidence for staging environment
./generate-backup-evidence.sh staging
```

**Note**: The script automatically detects whether backup infrastructure is enabled and adjusts evidence collection accordingly.

### Step 3: Review Generated Evidence

The script creates evidence in `backup-isolation-evidence/[timestamp]/`:

#### 📄 Evidence Files Generated:

**When Backup Infrastructure is ENABLED:**
1. **SOC2-Recovery-Data-Isolation-Evidence.md** - Full compliance evidence
2. **isolation-validation.md** - Geographic and access control validation
3. **terraform-outputs.md** - Complete infrastructure configuration
4. **gcp-storage-evidence.json** - Production, backup, and archive buckets
5. **service-account-permissions.json** - Dedicated backup service accounts
6. **access-control-tests.md** - Multi-bucket IAM validation

**When Backup Infrastructure is DISABLED:**
1. **SOC2-Recovery-Data-Isolation-Evidence.md** - Configuration status and rationale
2. **isolation-validation.md** - Explains disabled state is expected
3. **terraform-outputs.md** - Shows backup configuration status
4. **gcp-storage-evidence.json** - Production buckets only
5. **sql-backup-config.json** - Database backup configuration (always present)

### Step 4: Production Evidence Collection

For **production environments**, collect evidence showing full backup isolation:

```bash
# Production evidence (backup infrastructure auto-enabled)
./generate-backup-evidence.sh prod

# This will generate complete evidence showing:
# ✅ Cross-region backup storage
# ✅ Dedicated backup service accounts
# ✅ Database replicas in separate regions
# ✅ Geographic isolation validation
```

### Step 5: Take Screenshots for Visual Evidence

#### 5.1 GCP Console Screenshots

**If Backup Infrastructure is ENABLED:**
1. Go to GCP Console → Cloud Storage
2. Take screenshot showing:
   - Production bucket location (e.g., `us-west1`)
   - Backup bucket location (e.g., `us-central1`) 
   - Archive bucket location (e.g., `us-east1`)

**Service Account Permissions:**
1. Go to GCP Console → IAM & Admin → Service Accounts
2. Take screenshot of backup service account with restricted permissions
3. Show IAM policy differences between production and backup buckets

**If Backup Infrastructure is DISABLED:**
1. Show Terraform configuration demonstrating optional nature
2. Screenshot showing cost optimization for non-production environments
3. Document configuration variables and their values

#### 5.2 Command Line Evidence

```bash
# Show backup configuration status
terraform output backup_configuration

# If backup infrastructure is enabled:
gcloud storage buckets list --format="table(name,location,storageClass)"
gcloud iam service-accounts list --filter="displayName:backup"

# Show cross-region SQL instances (if enabled)
gcloud sql instances list --format="table(name,region,databaseVersion,status)"
```

### Step 6: Configuration Evidence

#### 6.1 Document Variable Configuration

Create a file showing your backup configuration:

```bash
# Show current backup configuration
cat > backup-configuration-evidence.md << EOF
# Backup Infrastructure Configuration

## Current Settings
- **Workspace**: $(terraform workspace show)
- **Backup Enabled**: $(terraform output -json backup_configuration | jq -r '.enabled')
- **Backup Region**: $(terraform output -json backup_configuration | jq -r '.backup_region')
- **Archive Region**: $(terraform output -json backup_configuration | jq -r '.archive_region')

## Policy Settings
- **Backup Retention**: $(terraform output -json backup_configuration | jq -r '.backup_retention_days') days
- **Archive After**: $(terraform output -json backup_configuration | jq -r '.archive_after_days') days
- **Archive Retention**: $(terraform output -json backup_configuration | jq -r '.archive_retention_days') days

## Configuration Rationale
- Production environments: Full backup isolation enabled
- Non-production environments: Optional (cost optimization)
- All environments: Database backups always enabled
EOF
```

### Step 7: Multi-Environment Evidence

For comprehensive SOC2 evidence, collect from multiple environments:

```bash
# Collect evidence from production (full backup isolation)
./generate-backup-evidence.sh prod

# Collect evidence from staging (showing configuration flexibility)
./generate-backup-evidence.sh staging

# Compare configurations
diff backup-isolation-evidence/*/backup-configuration-evidence.md
```

### Step 8: Package Evidence for Auditors

The script creates environment-aware evidence packages:

```bash
# Evidence shows different configurations appropriately
ls -la backup-isolation-evidence/backup-isolation-evidence-*.tar.gz
```

## 📊 Evidence Validation Checklist

### For Production Environments:
- [ ] **Geographic Separation**: Backup storage in different regions than production
- [ ] **Access Controls**: Dedicated service accounts with restricted permissions
- [ ] **Database Replicas**: Cross-region database replicas configured
- [ ] **Monitoring**: Backup failure alerting configured and tested
- [ ] **Retention Policies**: Appropriate backup and archive retention

### For Non-Production Environments:
- [ ] **Configuration Documentation**: Shows backup infrastructure is optional
- [ ] **Cost Optimization**: Demonstrates efficient resource allocation
- [ ] **Capability Demonstration**: Shows backup infrastructure can be enabled
- [ ] **Database Backups**: Basic backup capabilities still functional

## 🔧 Advanced Configuration Examples

### Custom Regional Configuration

```bash
# Deploy backup infrastructure to specific regions
./tf-workspace.sh prod apply \
  -var="backup_region=europe-west1" \
  -var="archive_region=asia-northeast1"
```

### Extended Retention Policies

```bash
# Configure longer retention for compliance
./tf-workspace.sh prod apply \
  -var="backup_retention_days=365" \
  -var="archive_retention_days=3653"  # 10 years
```

### Development Environment Testing

```bash
# Enable full backup infrastructure in dev for testing
./tf-workspace.sh dev apply \
  -var="enable_backup_infrastructure=true" \
  -var="backup_region=us-west2" \
  -var="archive_region=us-east4"
```

## 🔄 Ongoing Compliance

### Quarterly Evidence Collection

```bash
# Automated collection for all environments
for env in prod staging dev; do
  ./generate-backup-evidence.sh $env
done
```

### Configuration Validation

```bash
# Validate backup configuration across environments
terraform workspace select prod
terraform output backup_configuration

terraform workspace select staging  
terraform output backup_configuration
```

## 🚨 Important Notes

### Environment-Specific Behavior

- **Production**: Backup infrastructure **automatically enabled**
- **Staging/Dev**: Backup infrastructure **optional** (configurable)
- **All Environments**: Database backups always configured

### Cost Considerations

- **Full isolation**: Higher cost (3 regions, dedicated accounts, monitoring)
- **Selective deployment**: Cost-effective for non-production
- **Configurable retention**: Optimize storage costs

### Compliance Strategy

- **Production protection**: Complete isolation for critical data
- **Flexible deployment**: Enable isolation where required
- **Evidence completeness**: Document both enabled and disabled states
- **Cost justification**: Show appropriate resource allocation

## 🔗 Related Documentation

- [Terraform Variables](backup-isolation.tf) - Complete variable documentation
- [DR Testing README](DR_TESTING_README.md) - Disaster recovery testing procedures
- [Infrastructure Configuration](main.tf) - Base infrastructure documentation

---

**Key Insight**: The optional backup infrastructure demonstrates both **strong security practices** (when enabled) and **cost optimization** (when disabled), providing auditors with evidence of thoughtful resource management while maintaining the capability to achieve full compliance when required. 