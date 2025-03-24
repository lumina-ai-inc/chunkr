# GitHub Workflows

This directory contains GitHub Actions workflows that automate various aspects of the development, testing, and release processes.

## Workflows Overview

| Workflow | File | Description |
|----------|------|-------------|
| Release Please | [release-please.yml](./workflows/release-please.yml) | Manages releases, builds Docker images, publishes Helm charts, and updates compose files |
| Rust Lint | [rust-lint.yml](./workflows/rust-lint.yml) | Performs linting and formatting checks for Rust code |
| TypeScript Tests | [typescript-tests.yml](./workflows/typescript-tests.yml) | Runs build and lint tests for TypeScript code |

## Detailed Workflow Descriptions

### Release Please

This workflow automates the release process using Google's Release Please tool. It:

1. Creates release PRs with changelog entries when triggered by pushes to main
2. Builds and publishes Docker images for server, task, web, and doctr components
3. Packages and publishes Helm charts
4. Updates compose files with new version numbers
5. Publishes the Python client to PyPI when relevant files change

**Triggers:**
- Push to `main` branch (except changes to compose.yaml/compose-cpu.yaml)
- Manual workflow dispatch with a specified tag

### Rust Lint

This workflow ensures Rust code quality through:

1. Code formatting verification with `rustfmt`
2. Linting with `clippy`
3. Build verification with `cargo check`
4. Automatic PR creation for code formatting and clippy fixes

**Triggers:**
- Push to `main` branch affecting Rust files
- Pull requests to `main` branch affecting Rust files
- Manual workflow dispatch

### TypeScript Tests

This workflow tests TypeScript code in the web application by:

1. Building the web application
2. Running linting checks

**Triggers:**
- Pull requests

## Testing Workflows Manually

You can manually trigger workflows using GitHub's UI or the GitHub CLI.

### Using GitHub CLI

To run a workflow with the GitHub CLI, use the following command format:

```bash
gh workflow run <workflow-name> --ref <branch-name>
```

Replace `<workflow-name>` with the name of the workflow you want to run, and `<branch-name>` with the name of the branch you want to run the workflow on.

#### Examples:

**Release Please workflow:**

```bash
gh workflow run "Release Please" --ref main -F tag=1.6.1
```

**Rust Lint workflow:** 

```bash
gh workflow run "Rust Lint" --ref main
```

**TypeScript Tests workflow:**

```bash
gh workflow run "Typescript SDK test suite" --ref main
```

### Using GitHub UI

1. Navigate to the "Actions" tab in your GitHub repository
2. Select the workflow you want to run
3. Click "Run workflow" dropdown
4. Select the branch and fill in any required inputs
5. Click "Run workflow"