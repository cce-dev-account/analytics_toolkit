# Release Checklist

Use this checklist to ensure all steps are completed before creating a release.

## 🚀 Pre-Release Checklist

### Repository Setup
- [ ] All secrets are configured in GitHub repository:
  - [ ] `PYPI_API_TOKEN` (for PyPI publishing)
  - [ ] `TEST_PYPI_API_TOKEN` (for Test PyPI publishing)
- [ ] GitHub environments are set up (optional but recommended):
  - [ ] `production` environment with required reviewers
  - [ ] `test-pypi` environment for pre-releases
- [ ] Branch protection rules are configured for `main` branch

### Code Quality
- [ ] All tests pass locally: `./scripts/dev.sh test`
- [ ] Code is properly formatted: `./scripts/dev.sh format`
- [ ] Linting passes: `./scripts/dev.sh lint`
- [ ] Type checking passes: `./scripts/dev.sh typecheck`
- [ ] Security scans pass: `./scripts/dev.sh security`
- [ ] No TODO or FIXME comments in critical code paths
- [ ] Documentation is up to date

### Version Management
- [ ] Decide on version number following [Semantic Versioning](https://semver.org/):
  - **Patch** (1.0.1): Bug fixes only
  - **Minor** (1.1.0): New features, backward compatible
  - **Major** (2.0.0): Breaking changes
  - **Pre-release** (1.0.0-alpha.1): Alpha, beta, or release candidate
- [ ] Update version in `pyproject.toml` if doing manual release
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) for better changelog

### Git Preparation
- [ ] Working directory is clean (no uncommitted changes)
- [ ] Currently on `main` or `master` branch
- [ ] Local branch is up to date with remote: `git pull origin main`
- [ ] All intended changes are merged into main branch

## 📋 Release Process

### Option 1: Automated Release (Recommended)

#### Using Release Script
```bash
# Run all checks and prepare release
./scripts/release.sh release 1.0.0

# Or step by step:
./scripts/release.sh check           # Run pre-release checks
./scripts/release.sh prepare 1.0.0   # Prepare release
./scripts/release.sh tag 1.0.0       # Create and push tag
```

#### Manual Tag Creation
```bash
# Create and push version tag
git tag v1.0.0
git push origin v1.0.0
```

### Option 2: Manual Release (GitHub UI)

1. Go to repository `Actions` tab
2. Select "Release Pipeline" workflow
3. Click "Run workflow"
4. Enter version number (e.g., `1.0.0`)
5. Check "pre-release" if applicable
6. Click "Run workflow"

## 🔍 Post-Release Verification

### GitHub Actions
- [ ] Release workflow completed successfully
- [ ] All jobs passed:
  - [ ] Validate Release
  - [ ] Generate Changelog
  - [ ] Publish to PyPI
  - [ ] Create GitHub Release
  - [ ] Update Documentation (for stable releases)

### PyPI
- [ ] Package is available on PyPI: `https://pypi.org/project/analytics-toolkit/`
- [ ] Package version matches released version
- [ ] Package can be installed: `pip install analytics-toolkit==1.0.0`
- [ ] Package imports correctly: `python -c "import analytics_toolkit; print(analytics_toolkit.__version__)"`

### GitHub Release
- [ ] GitHub release is created: `https://github.com/your-org/analytics-toolkit/releases`
- [ ] Release notes are properly generated
- [ ] Release assets (wheel and source) are attached
- [ ] Pre-release flag is set correctly (if applicable)

### Documentation
- [ ] Documentation is updated (for stable releases)
- [ ] Version numbers in docs match release
- [ ] Installation instructions are correct

## 🚨 Rollback Procedure

If something goes wrong with the release:

### 1. Quick Fixes
- [ ] Fix issues and create a new patch release
- [ ] Use hotfix process: `./scripts/release.sh hotfix 1.0.1`

### 2. Remove from PyPI (Last Resort)
- [ ] Contact PyPI support to remove package version
- [ ] Note: PyPI generally doesn't allow re-uploading same version

### 3. GitHub Release
- [ ] Edit or delete GitHub release if needed
- [ ] Update release notes with correction information

## 📝 Release Types

### Stable Release
- Version: `1.0.0`
- Publishes to: PyPI
- Creates: GitHub Release
- Updates: Documentation

### Pre-Release
- Version: `1.0.0-alpha.1`, `1.0.0-beta.1`, `1.0.0-rc.1`
- Publishes to: Test PyPI
- Creates: GitHub Pre-Release
- Updates: No documentation

### Hotfix Release
- Version: `1.0.1` (patch increment)
- Process: Same as stable release
- Purpose: Critical bug fixes only

## 🔧 Common Issues

### Authentication Errors
- [ ] Verify PyPI tokens are correct and not expired
- [ ] Check environment setup in GitHub settings
- [ ] Ensure tokens have appropriate scope

### Version Conflicts
- [ ] Check if version already exists on PyPI
- [ ] Increment version number if needed
- [ ] Delete and recreate git tag if necessary

### Test Failures
- [ ] Run tests locally to identify issues
- [ ] Fix failing tests before retrying release
- [ ] Ensure all dependencies are properly locked

### Build Errors
- [ ] Verify `pyproject.toml` is valid
- [ ] Check Poetry lock file is up to date
- [ ] Test build locally: `poetry build`

## 📞 Getting Help

If you encounter issues during release:

1. Check the [Release Documentation](docs/RELEASE.md)
2. Review GitHub Actions logs for detailed error messages
3. Test the release process locally using the release script
4. Consult the troubleshooting section in the release docs
5. Open an issue with full error details and context

---

**Remember:** Releases are permanent on PyPI. Take time to verify everything is correct before proceeding.