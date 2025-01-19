# Syncing Public Repo with Private Enterprise Repo

## Initial Setup
1. Clone the private enterprise repository:
```bash
git clone https://github.com/lumina-ai-inc/chunkr-enterprise.git
cd chunkr-enterprise
```

2. Add the public repo as a remote source:
```bash
git remote add public https://github.com/lumina-ai-inc/chunkr.git
```

## Pulling Updates
To get the latest changes from the public repository:
1. Fetch the latest changes:
```bash
git fetch public
```

2. Merge the changes into your main branch:
```bash
git merge public/main
```

## Verify Setup
To verify your remotes are configured correctly:
```bash
git remote -v
```
Should show both `origin` (private) and `public` remotes.

## Troubleshooting
If you encounter fetch errors, try:
```bash
git config --global http.postBuffer 524288000
```
or
```bash
git config --global http.version HTTP/1.1
```

