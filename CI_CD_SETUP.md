# GitHub Actions CI/CD Pipeline Setup Guide

This guide explains how to configure the GitHub Actions pipeline for your sentiment analysis MLOps project.

## 🎯 Pipeline Overview

The CI/CD pipeline automates the deployment process in 10 stages:

```
Training (local) → Pull Model → Build → Lint → Test → Push DockerHub 
→ Deploy Test → Smoke Tests → Manual Approval → Tag Prod → Deploy Prod
```

### Stage Breakdown:

1. **Training (Local)**: Run `dvc repro` locally to train all models and register best model to MLflow
2. **Pull Model**: Download production model from MLflow Model Registry
3. **Build**: Build Docker images for backend and frontend
4. **Lint**: Run flake8, black, isort to check code quality
5. **Test**: Run pytest unit tests with PostgreSQL service
6. **Push DockerHub**: Push images with `:test` tag to Docker Hub
7. **Deploy Test**: Deploy to test EC2 environment
8. **Smoke Tests**: Verify deployment with health checks and sample predictions
9. **Manual Approval**: ⏸️ Wait for human approval in GitHub UI
10. **Tag Prod**: Retag images as `:prod`
11. **Deploy Prod**: Deploy to production EC2 environment

---

## 📋 Prerequisites

### 1. Local Training Complete

Before running the pipeline, train models locally:

```bash
# Run DVC pipeline to train all models
dvc repro

# This will:
# - Train SVM baseline (no SMOTE)
# - Train SVM with SMOTE
# - Fine-tune DistilBERT
# - Evaluate all models and register best to MLflow
```

### 2. Model Registered in MLflow

After training, promote the best model to production:

```bash
# Check available models
python scripts/promote_model.py --latest --alias staging

# Promote to production (required for CI/CD)
python scripts/promote_model.py --latest --alias production
```

### 3. GitHub Repository Setup

Push your code to GitHub:

```bash
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
```

---

## 🔐 Required GitHub Secrets

Go to: **Settings → Secrets and variables → Actions → New repository secret**

Add the following secrets:

### MLflow / DagsHub Credentials

| Secret Name | Description | Example |
|------------|-------------|---------|
| `DAGSHUB_USERNAME` | Your DagsHub username | `sairam030` |
| `DAGSHUB_TOKEN` | DagsHub personal access token | Generate at: https://dagshub.com/user/settings/tokens |

### Docker Hub Credentials

| Secret Name | Description | Example |
|------------|-------------|---------|
| `DOCKERHUB_TOKEN` | Docker Hub access token | Generate at: https://hub.docker.com/settings/security |

Note: `DOCKERHUB_USERNAME` is set in workflow as `sairam030` (change if needed)

### Test Environment (EC2)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `TEST_EC2_HOST` | Public IP of test EC2 | `18.61.98.223` |
| `TEST_EC2_USER` | SSH username | `ubuntu` or `ec2-user` |
| `TEST_EC2_SSH_KEY` | Private SSH key (full content) | `-----BEGIN RSA PRIVATE KEY-----\n...` |

### Production Environment (EC2)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `PROD_EC2_HOST` | Public IP of prod EC2 | `52.15.123.45` |
| `PROD_EC2_USER` | SSH username | `ubuntu` or `ec2-user` |
| `PROD_EC2_SSH_KEY` | Private SSH key (full content) | `-----BEGIN RSA PRIVATE KEY-----\n...` |
| `PROD_BACKEND_HOST` | Private IP of prod backend EC2 | `10.0.1.50` |

### Database (RDS)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `RDS_DATABASE_URL` | Full PostgreSQL connection URL | `postgresql://sentiment_user:PASSWORD@sentiment-db.cv2iq0sos7cc.ap-south-2.rds.amazonaws.com:5432/sentiment_db` |

---

## ⚙️ Setting Up Manual Approval

The pipeline includes a **manual approval gate** between test and production deployment.

### Step 1: Create Production Environment

1. Go to: **Settings → Environments**
2. Click **New environment**
3. Name: `production-approval`
4. Click **Add environment**

### Step 2: Add Protection Rules

1. Check **Required reviewers**
2. Add yourself (or team members) as reviewers
3. Optionally set **Wait timer** (e.g., 5 minutes minimum wait)
4. Click **Save protection rules**

### Step 3: How It Works

When the pipeline reaches the approval stage:
1. You'll receive a GitHub notification
2. Go to: **Actions → Deploy Pipeline → Review pending deployments**
3. Review test environment: http://TEST_EC2_HOST/dashboard
4. If everything looks good, click **Approve and deploy**
5. If issues found, click **Reject** and fix the problems

---

## 🚀 Running the Pipeline

### Automatic Trigger

The pipeline runs automatically on every push to `main`:

```bash
git add .
git commit -m "Update model training"
git push origin main
```

### Manual Trigger

You can also trigger manually:

1. Go to **Actions** tab
2. Click **MLOps CI/CD Pipeline**
3. Click **Run workflow**
4. Select branch (`main`)
5. Click **Run workflow**

---

## 📊 Monitoring Pipeline Execution

### View Pipeline Status

1. Go to **Actions** tab
2. Click on latest workflow run
3. View stage-by-stage progress

### Stage Details

Click on any stage to see:
- Command output
- Error messages (if failed)
- Artifact downloads
- Test results

### Common Issues

#### ❌ Pull Model Failed
- **Cause**: MLflow credentials incorrect or model not promoted
- **Fix**: Check `DAGSHUB_TOKEN` and run `python scripts/promote_model.py --latest --alias production`

#### ❌ Lint Failed
- **Cause**: Code doesn't meet PEP8 standards
- **Fix**: Run locally: `flake8 aws_serving/app/` and `black aws_serving/app/`

#### ❌ Tests Failed
- **Cause**: Unit tests failing
- **Fix**: Run locally: `cd aws_serving/app && pytest ../../tests/`

#### ❌ Deploy Test Failed
- **Cause**: SSH connection issues or Docker Compose errors
- **Fix**: Verify EC2 security groups allow SSH (port 22) and HTTP (port 80)

#### ❌ Smoke Tests Failed
- **Cause**: Application not responding correctly
- **Fix**: SSH into test EC2, check logs: `docker-compose logs backend`

---

## 🔧 Customization

### Change Docker Registry

Edit [.github/workflows/deploy.yml](.github/workflows/deploy.yml):

```yaml
env:
  DOCKER_REGISTRY: docker.io
  DOCKERHUB_USERNAME: your-username  # Change this
```

### Change Python Version

Edit build steps:

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'  # Change from 3.10
```

### Add More Tests

Create additional test files in `tests/` directory:

```python
# tests/test_new_feature.py
def test_new_endpoint(client):
    response = client.get('/api/new-endpoint')
    assert response.status_code == 200
```

### Skip Stages

Add `if: false` to skip a stage:

```yaml
lint:
  name: 🔍 Lint & Code Quality
  if: false  # Skip this stage
  runs-on: ubuntu-latest
```

---

## 📚 Understanding the Workflow

### Why Manual Approval?

Manual approval ensures:
- **Human Review**: Someone verifies test deployment before production
- **Business Logic**: Check metrics in dashboard before promoting
- **Safety**: Prevent automatic deployment of buggy code

### Why Separate Test/Prod Tags?

- `:test` tag → Used in test environment
- `:prod` tag → Used in production environment
- Allows **rollback**: If prod fails, redeploy previous `:prod` tag
- Enables **testing**: Test changes thoroughly before production

### Why Download Model from MLflow?

- **Single Source of Truth**: MLflow stores the best model
- **Reproducibility**: Know exactly which model version is deployed
- **Comparison**: MLflow tracks all experiments and metrics
- **No Model in Git**: Large model files don't belong in Git repos

---

## 🧪 Testing Locally

Before pushing to GitHub, test components locally:

### 1. Test Model Download

```bash
python scripts/pull_best_model.py --alias production --output /tmp/test-artifacts
ls -lh /tmp/test-artifacts/
```

### 2. Test Docker Build

```bash
cd aws_serving/app
docker build -t test-backend .
docker run -p 5000:5000 test-backend
```

### 3. Test Smoke Tests

```bash
./scripts/smoke_test.sh http://localhost
```

### 4. Test Linting

```bash
flake8 aws_serving/app/ --max-line-length=120
black --check aws_serving/app/
isort --check-only aws_serving/app/
```

### 5. Test Unit Tests

```bash
cd aws_serving/app
pytest ../../tests/ -v
```

---

## 📖 Pipeline Flow Diagram

```
┌─────────────────────┐
│   LOCAL MACHINE     │
│                     │
│  dvc repro          │ ← Train models
│  promote to prod    │ ← Register best model
└──────────┬──────────┘
           │ git push
           ↓
┌─────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                          │
│  │ Pull Model   │ Download from MLflow                     │
│  └──────┬───────┘                                          │
│         ↓                                                   │
│  ┌──────────────┐                                          │
│  │ Build        │ Build Docker images                      │
│  └──────┬───────┘                                          │
│         ↓                                                   │
│  ┌──────────────┐    ┌──────────────┐                     │
│  │ Lint         │    │ Test         │                      │
│  └──────┬───────┘    └──────┬───────┘                     │
│         └────────┬───────────┘                             │
│                  ↓                                          │
│  ┌──────────────────────────┐                             │
│  │ Push to DockerHub :test  │                             │
│  └──────────────┬───────────┘                             │
│                 ↓                                           │
│  ┌──────────────────────────┐                             │
│  │ Deploy to Test EC2       │                             │
│  └──────────────┬───────────┘                             │
│                 ↓                                           │
│  ┌──────────────────────────┐                             │
│  │ Smoke Tests              │                              │
│  └──────────────┬───────────┘                             │
│                 ↓                                           │
│  ╔══════════════════════════╗                             │
│  ║  ⏸️  MANUAL APPROVAL     ║ ← Review dashboard         │
│  ╚══════════════╤═══════════╝   Click approve button      │
│                 ↓                                           │
│  ┌──────────────────────────┐                             │
│  │ Tag as :prod             │                              │
│  └──────────────┬───────────┘                             │
│                 ↓                                           │
│  ┌──────────────────────────┐                             │
│  │ Deploy to Prod EC2       │                              │
│  └──────────────────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Learning Resources

### GitHub Actions
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Environment Protection Rules](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)

### MLOps Best Practices
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DVC Pipelines](https://dvc.org/doc/start/data-management/data-pipelines)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)

### Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [Flake8 Rules](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- [Black Formatter](https://black.readthedocs.io/)

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Pipeline Logs**: Actions tab → Click failed stage → Read output
2. **Test Locally**: Run commands locally before pushing
3. **Verify Secrets**: Ensure all GitHub secrets are set correctly
4. **Check EC2**: SSH into EC2 and verify Docker is running
5. **Review Dashboard**: Check test environment dashboard for errors

---

## ✅ Next Steps

After pipeline setup:

1. ✅ Configure all GitHub secrets
2. ✅ Set up production environment with approval rules
3. ✅ Train models locally: `dvc repro`
4. ✅ Promote best model: `python scripts/promote_model.py --latest --alias production`
5. ✅ Push to GitHub: `git push origin main`
6. ✅ Monitor pipeline in Actions tab
7. ✅ Review test deployment
8. ✅ Approve production deployment
9. ✅ Verify production dashboard

**Congratulations! Your MLOps CI/CD pipeline is now operational! 🎉**
