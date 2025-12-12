# Contributing to PIVOT

## Commit Message Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scopes

Common scopes for this project:
- **data**: Data preprocessing, loading, augmentation
- **model**: Model architecture, layers, heads
- **train**: Training pipeline, loops, callbacks
- **loss**: Loss functions
- **inference**: Inference pipeline, post-processing
- **api**: FastAPI server, endpoints
- **viz**: Visualization, graphs, dashboards
- **eval**: Evaluation metrics, validation
- **deploy**: Deployment, Docker, infrastructure
- **monitor**: Monitoring, logging, alerts

### Examples

```
feat[data]: add LUNA16 DICOM loader
fix[train]: resolve NaN loss issue in multi-task learning
docs[api]: update API endpoint documentation
refactor[model]: simplify U-Net decoder architecture
perf[inference]: optimize sliding window overlap computation
test[loss]: add unit tests for focal loss
chore[deps]: update PyTorch to 2.1.0
```

### Breaking Changes

Breaking changes should be indicated by a `!` after the type/scope and explained in the footer:

```
feat[api]!: change response format for /metrics endpoint

BREAKING CHANGE: The /metrics endpoint now returns data in a different JSON structure.
Clients need to update their parsing logic.
```

## Setting Up Pre-commit

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks:
   ```bash
   pre-commit install --hook-type commit-msg
   pre-commit install  # for other hooks
   ```

3. (Optional) Run against all files:
   ```bash
   pre-commit run --all-files
   ```

## Commit Message Validation

The pre-commit hook will automatically validate your commit messages. If your commit message doesn't follow the conventional format, the commit will be rejected with an error message.

### Valid Examples
✅ `feat[data]: implement isotropic resampling`
✅ `fix: resolve GPU memory overflow`
✅ `docs: update README with setup instructions`
✅ `refactor[model]!: redesign detection head architecture`

### Invalid Examples
❌ `added new feature` (missing type)
❌ `feat add feature` (missing colon)
❌ `FIX[data]: bug fix` (type must be lowercase)
❌ `feat[data] : fix` (space before colon)

## Bypassing the Hook (Not Recommended)

In exceptional cases, you can bypass the hook with:
```bash
git commit --no-verify -m "your message"
```

However, this should be avoided as it breaks the commit convention.
