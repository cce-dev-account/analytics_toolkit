# Release Documentation

This document provides instructions for setting up automated releases for the Analytics Toolkit project.

## 🔐 Setting Up Secrets

### Required Secrets

The automated release workflow requires the following secrets to be configured in your GitHub repository:

#### 1. PyPI API Token (`PYPI_API_TOKEN`)

**Required for:** Publishing packages to PyPI

**Setup Steps:**

1. **Create PyPI Account:**
   - Go to [https://pypi.org/account/register/](https://pypi.org/account/register/)
   - Register for a new account or log into existing account

2. **Generate API Token:**
   - Go to [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
   - Click "Add API token"
   - **Token name:** `analytics-toolkit-github-actions`
   - **Scope:** Select "Entire account" or limit to specific project
   - Click "Add token"
   - **⚠️ IMPORTANT:** Copy the token immediately (starts with `pypi-`)

3. **Add to GitHub Secrets:**
   - Go to your repository: `Settings` → `Secrets and variables` → `Actions`
   - Click "New repository secret"
   - **Name:** `PYPI_API_TOKEN`
   - **Value:** Paste the PyPI token (including `pypi-` prefix)
   - Click "Add secret"

#### 2. Test PyPI API Token (`TEST_PYPI_API_TOKEN`)

**Required for:** Publishing pre-release packages to Test PyPI

**Setup Steps:**

1. **Create Test PyPI Account:**
   - Go to [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)
   - Register for a new account (separate from main PyPI)

2. **Generate API Token:**
   - Go to [https://test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)
   - Click "Add API token"
   - **Token name:** `analytics-toolkit-github-actions-test`
   - **Scope:** Select "Entire account"
   - Click "Add token"
   - Copy the token (starts with `pypi-`)

3. **Add to GitHub Secrets:**
   - Go to your repository: `Settings` → `Secrets and variables` → `Actions`
   - Click "New repository secret"
   - **Name:** `TEST_PYPI_API_TOKEN`
   - **Value:** Paste the Test PyPI token
   - Click "Add secret"

### 🔒 Environment Setup

For additional security, it's recommended to use GitHub Environments:

#### 1. Create Production Environment

1. Go to `Settings` → `Environments`
2. Click "New environment"
3. **Name:** `production`
4. **Environment protection rules:**
   - ☑️ Required reviewers (add team members)
   - ☑️ Wait timer: 5 minutes (optional)
   - ☑️ Restrict pushes to protected branches
5. **Environment secrets:**
   - Add `PYPI_API_TOKEN` here instead of repository secrets for extra security

#### 2. Create Test Environment

1. Click "New environment"
2. **Name:** `test-pypi`
3. **Environment secrets:**
   - Add `TEST_PYPI_API_TOKEN`

### 📋 Verification Checklist

Before creating your first release, verify:

- [ ] `PYPI_API_TOKEN` secret is configured
- [ ] `TEST_PYPI_API_TOKEN` secret is configured
- [ ] Repository has appropriate branch protection rules
- [ ] PyPI project name matches your package name
- [ ] All tests pass in CI

## 🚀 Release Process

### Automatic Release (Recommended)

#### 1. Create and Push Version Tag

```bash
# Update version in pyproject.toml (optional - workflow will do this)
poetry version 1.0.0

# Commit version bump (optional)
git add pyproject.toml
git commit -m "bump: version 1.0.0"

# Create and push tag
git tag v1.0.0
git push origin v1.0.0
```

#### 2. Monitor Release Workflow

1. Go to `Actions` tab in your repository
2. Watch the "Release Pipeline" workflow
3. The workflow will:
   - ✅ Validate the release version
   - ✅ Run full test suite
   - ✅ Run code quality checks
   - ✅ Build the package
   - ✅ Generate changelog from commits
   - ✅ Publish to PyPI
   - ✅ Create GitHub release with assets
   - ✅ Update documentation

### Manual Release

You can also trigger releases manually:

1. Go to `Actions` → `Release Pipeline`
2. Click "Run workflow"
3. **Branch:** Select `main`
4. **Version:** Enter version (e.g., `1.0.0`)
5. **Pre-release:** Check if this is a pre-release
6. Click "Run workflow"

### Pre-release Process

For pre-releases (alpha, beta, rc):

```bash
# Create pre-release tag
git tag v1.0.0-alpha.1
git push origin v1.0.0-alpha.1
```

Pre-releases will:
- ✅ Publish to Test PyPI instead of main PyPI
- ✅ Mark GitHub release as "pre-release"
- ❌ Skip documentation deployment

### Version Naming Convention

Follow [Semantic Versioning](https://semver.org/):

- **Patch release:** `v1.0.1` (bug fixes)
- **Minor release:** `v1.1.0` (new features, backward compatible)
- **Major release:** `v2.0.0` (breaking changes)
- **Pre-release:** `v1.0.0-alpha.1`, `v1.0.0-beta.1`, `v1.0.0-rc.1`

## 📝 Changelog Generation

The workflow automatically generates changelogs based on commit messages. Use conventional commits for better changelog organization:

```bash
# Features
git commit -m "feat: add new data preprocessing pipeline"

# Bug fixes
git commit -m "fix: resolve memory leak in model training"

# Documentation
git commit -m "docs: update API documentation"

# Other changes
git commit -m "refactor: improve code organization"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. PyPI Authentication Failed

**Error:** `HTTP Error 403: Invalid or non-existent authentication information`

**Solutions:**
- Verify `PYPI_API_TOKEN` secret is correctly set
- Check token has appropriate permissions
- Ensure token hasn't expired
- Try regenerating the token

#### 2. Package Already Exists

**Error:** `File already exists`

**Solutions:**
- PyPI doesn't allow re-uploading same version
- Increment version number
- Delete the tag and create a new one with higher version

#### 3. Tests Failing in Release

**Error:** Test failures during release validation

**Solutions:**
- Run tests locally: `./scripts/dev.sh test`
- Fix failing tests
- Push fixes before creating release tag

#### 4. Version Mismatch

**Error:** Version in tag doesn't match expected format

**Solutions:**
- Use format: `v1.0.0` (with 'v' prefix)
- Follow semantic versioning
- Avoid special characters in version

### Debug Commands

```bash
# Test package locally
poetry build
twine check dist/*

# Test PyPI upload (dry run)
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry publish --repository test-pypi --dry-run

# Verify package installation
pip install analytics-toolkit==1.0.0
python -c "import analytics_toolkit; print(analytics_toolkit.__version__)"
```

## 📚 Additional Resources

- [Poetry Publishing Documentation](https://python-poetry.org/docs/repositories/)
- [PyPI API Tokens Guide](https://pypi.org/help/#apitoken)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## 🆘 Getting Help

If you encounter issues with the release process:

1. Check the GitHub Actions logs for detailed error messages
2. Verify all secrets are correctly configured
3. Test the package build locally
4. Consult the troubleshooting section above
5. Open an issue in the repository with full error details