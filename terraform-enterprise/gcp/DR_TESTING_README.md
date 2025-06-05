# Disaster Recovery Testing for SOC2 Compliance

This directory contains tools for performing actual disaster recovery testing that validates your ability to recreate infrastructure from scratch.

## 🚨 IMPORTANT SAFETY WARNINGS

- **NEVER run DR tests on production workspaces (`prod`, `turbolearn`)**
- Always use `dev` or `staging` workspaces for DR testing
- The DR test script **DESTROYS ALL INFRASTRUCTURE** in the specified workspace
- Only run these tests when you can afford to lose the test environment temporarily

## Quick Start

### 1. Test Your Current Setup (Safe)
```bash
# Test that everything is configured correctly
./tf-workspace.sh staging status
./dr-test.sh --help
```

### 2. Run Basic DR Test (Destructive)
```bash
# Basic DR test - destroys and recreates staging environment
./dr-test.sh staging
```

### 3. Run Cross-Region DR Test (Advanced)
```bash
# Test disaster recovery to a different region
./dr-test.sh staging --dr-region us-west2 --dr-zone c
```

## What This Actually Tests

This performs **real disaster recovery testing**:

1. **📊 Pre-Destruction State Capture**
   - Takes terraform plan snapshot
   - Captures all GCP resource states
   - Records current outputs and configurations

2. **💥 Complete Infrastructure Destruction**
   - Destroys ALL resources using `terraform destroy`
   - Verifies complete destruction
   - Measures destruction time

3. **🔄 Full Infrastructure Recreation**
   - Recreates entire infrastructure from scratch
   - Optionally in different region/zone (cross-region DR)
   - Measures recreation time

4. **✅ Validation & Testing**
   - Verifies terraform state matches desired state
   - Tests GKE cluster connectivity
   - Validates key services are accessible

## Files & Scripts

### Main Scripts
- `tf-workspace.sh` - Enhanced terraform wrapper with staging support and location overrides
- `dr-test.sh` - Main DR testing script that performs destruction/recreation
- `DR_TESTING_README.md` - This documentation

### Workspaces Supported
- `dev` - Development environment (safe for DR testing)
- `staging` - Staging environment (safe for DR testing) 
- `prod` - Production (BLOCKED from DR testing)
- `turbolearn` - Production (BLOCKED from DR testing)

## Script Features

### tf-workspace.sh Enhancements
- Added `staging` workspace support
- Location override options:
  - `--override-region <region>` - Test in different region
  - `--override-zone <zone_suffix>` - Test in different zone
  - `--override-project <project>` - Test in different project

### dr-test.sh Capabilities
- **Safety Features:**
  - Blocks production workspace usage
  - Requires explicit confirmation before destruction
  - Comprehensive logging of all operations

- **Testing Modes:**
  - Full DR test (destroy + recreate)
  - Destruction only (`--destroy-only`)
  - Recreation only (`--skip-destroy`)

- **Cross-Region Testing:**
  - `--dr-region` - Test recovery in different region
  - `--dr-zone` - Test recovery in different zone
  - `--dr-project` - Test recovery in different project

## Evidence Generated

All test runs create timestamped evidence in `dr-test-results/`:

### Logs
- `dr-test-TIMESTAMP.log` - High-level test execution log
- `dr-detailed-TIMESTAMP.log` - Detailed command output and errors

### State Captures
- `pre-destroy-outputs-TIMESTAMP.json` - Terraform outputs before destruction
- `post-recreation-outputs-TIMESTAMP.json` - Terraform outputs after recreation
- `*-instances-TIMESTAMP.json` - GCP compute instances before/after
- `*-sql-TIMESTAMP.json` - Cloud SQL instances before/after
- `*-buckets-TIMESTAMP.json` - Storage buckets before/after
- `*-clusters-TIMESTAMP.json` - GKE clusters before/after

### Performance Metrics
- `destruction-time-TIMESTAMP.txt` - How long destruction took
- `recreation-time-TIMESTAMP.txt` - How long recreation took

### Final Report
- `dr-test-report-TIMESTAMP.md` - Complete test report for SOC2 auditors

## Example SOC2 Evidence Structure

```
dr-test-results/
├── dr-test-report-20241201_143022.md      # Main audit evidence
├── dr-test-20241201_143022.log            # Test execution log
├── dr-detailed-20241201_143022.log        # Detailed technical log
├── destruction-time-20241201_143022.txt   # 245 seconds
├── recreation-time-20241201_143022.txt    # 890 seconds
└── [various JSON state files]             # GCP resource states
```

## Common Usage Patterns

### Quarterly DR Testing
```bash
# Run comprehensive DR test every quarter
./dr-test.sh staging
```

### Cross-Region DR Validation
```bash
# Test ability to recover in different region
./dr-test.sh staging --dr-region us-east1 --dr-zone b
```

### Test Script Development
```bash
# Test script changes without destruction
./dr-test.sh staging --skip-destroy

# Test destruction process only
./dr-test.sh staging --destroy-only
```

## SOC2 Compliance Value

This solution provides **actual evidence** of disaster recovery capability:

✅ **Real Recovery Testing** - Actually destroys and recreates infrastructure
✅ **Measurable RTO/RPO** - Provides exact recovery times
✅ **Cross-Region Validation** - Tests geographic disaster scenarios  
✅ **Complete Documentation** - Comprehensive logs and reports
✅ **Regular Testing Schedule** - Supports quarterly/annual testing
✅ **Audit Trail** - Timestamped evidence for compliance reviews

## Troubleshooting

### Common Issues

1. **Script Permission Denied**
```bash
chmod +x tf-workspace.sh dr-test.sh
```

2. **Doppler Authentication**
```bash
doppler login
doppler setup --project chunkr-infra
```

3. **GCloud Authentication**
```bash
gcloud auth login
gcloud config set project [your-project-id]
```

### Recovery from Failed Tests

If a DR test fails partway through:

1. Check the detailed log: `dr-test-results/dr-detailed-TIMESTAMP.log`
2. Manually run terraform commands to fix issues:
```bash
./tf-workspace.sh staging plan
./tf-workspace.sh staging apply
```

3. Re-run validation:
```bash
./dr-test.sh staging --skip-destroy
```

## Best Practices

1. **Schedule Regular Testing**
   - Run quarterly DR tests
   - Document in calendar with reminders

2. **Test Realistic Scenarios**
   - Use cross-region testing occasionally
   - Test during business hours to simulate real disasters

3. **Keep Evidence Organized**
   - Archive old test results
   - Maintain test result inventory

4. **Review and Improve**
   - Analyze recovery times for trends
   - Update procedures based on lessons learned
   - Share results with team

## Next Steps

1. Make scripts executable: `chmod +x *.sh`
2. Test in staging: `./dr-test.sh staging`
3. Schedule regular DR testing
4. Archive results for audit compliance 