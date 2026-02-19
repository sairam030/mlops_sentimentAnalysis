# GitHub Actions Environment Setup

## 🔐 Complete Secrets Configuration

Go to your GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**

---

## Required Secrets (12 Total)

### 1. MLflow / DagsHub (2 secrets)

```
Secret Name: DAGSHUB_USERNAME
Value: sairam030
```

```
Secret Name: DAGSHUB_TOKEN
How to get: https://dagshub.com/user/settings/tokens
Click "Generate new token" → Copy the token
Value: dags_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 2. Docker Hub (1 secret)

```
Secret Name: DOCKERHUB_TOKEN
How to get: https://hub.docker.com/settings/security
Click "New Access Token" → Name it "github-actions" → Copy token
Value: dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Note**: Username `sairam030` is hardcoded in workflow, no secret needed.

---

### 3. Test Environment - Frontend EC2 (3 secrets)

```
Secret Name: TEST_FRONTEND_HOST
Value: 18.61.98.223
```

```
Secret Name: TEST_FRONTEND_USER
Value: ubuntu
```

```
Secret Name: TEST_FRONTEND_SSH_KEY
How to get: cat ~/path/to/your-test-key.pem
Value: 
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(paste entire content of your .pem file)
...
-----END RSA PRIVATE KEY-----
```

---

### 4. Test Environment - Backend EC2 (3 secrets)

```
Secret Name: TEST_BACKEND_HOST
Value: 10.0.132.177
```

```
Secret Name: TEST_BACKEND_USER
Value: ubuntu
```

```
Secret Name: TEST_BACKEND_SSH_KEY
How to get: cat ~/path/to/your-test-key.pem
Value: 
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(same or different .pem file depending on your EC2 setup)
...
-----END RSA PRIVATE KEY-----
```

**Note**: If both EC2s use the same key pair, `TEST_FRONTEND_SSH_KEY` and `TEST_BACKEND_SSH_KEY` will have the same value.

---

### 5. Production Environment - Frontend EC2 (3 secrets)

```
Secret Name: PROD_FRONTEND_HOST
Value: <your-prod-frontend-public-ip>
Example: 52.66.123.45
```

```
Secret Name: PROD_FRONTEND_USER
Value: ubuntu
(or ec2-user for Amazon Linux)
```

```
Secret Name: PROD_FRONTEND_SSH_KEY
How to get: cat ~/path/to/your-prod-key.pem
Value: 
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(paste entire content of production .pem file)
...
-----END RSA PRIVATE KEY-----
```

---

### 6. Production Environment - Backend EC2 (3 secrets)

```
Secret Name: PROD_BACKEND_HOST
Value: <your-prod-backend-private-ip>
Example: 10.0.1.50
```

```
Secret Name: PROD_BACKEND_USER
Value: ubuntu
(or ec2-user for Amazon Linux)
```

```
Secret Name: PROD_BACKEND_SSH_KEY
How to get: cat ~/path/to/your-prod-key.pem
Value: 
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(same or different from frontend, depending on setup)
...
-----END RSA PRIVATE KEY-----
```

---

### 7. Database (1 secret)

```
Secret Name: RDS_DATABASE_URL
Value: postgresql://sentiment_user:<PASSWORD>@sentiment-db.cv2iq0sos7cc.ap-south-2.rds.amazonaws.com:5432/sentiment_db

Replace <PASSWORD> with your actual RDS password
```

---

## 📋 Quick Copy Template for GitHub UI

When adding secrets in GitHub, copy-paste this checklist:

### ✅ Core Services
- [ ] `DAGSHUB_USERNAME` = `sairam030`
- [ ] `DAGSHUB_TOKEN` = `dags_...`
- [ ] `DOCKERHUB_TOKEN` = `dckr_pat_...`

### ✅ Test Environment
- [ ] `TEST_FRONTEND_HOST` = `18.61.98.223`
- [ ] `TEST_FRONTEND_USER` = `ubuntu`
- [ ] `TEST_FRONTEND_SSH_KEY` = `-----BEGIN RSA PRIVATE KEY-----...`
- [ ] `TEST_BACKEND_HOST` = `10.0.132.177`
- [ ] `TEST_BACKEND_USER` = `ubuntu`
- [ ] `TEST_BACKEND_SSH_KEY` = `-----BEGIN RSA PRIVATE KEY-----...`

### ✅ Production Environment
- [ ] `PROD_FRONTEND_HOST` = `<your-ip>`
- [ ] `PROD_FRONTEND_USER` = `ubuntu`
- [ ] `PROD_FRONTEND_SSH_KEY` = `-----BEGIN RSA PRIVATE KEY-----...`
- [ ] `PROD_BACKEND_HOST` = `<your-private-ip>`
- [ ] `PROD_BACKEND_USER` = `ubuntu`
- [ ] `PROD_BACKEND_SSH_KEY` = `-----BEGIN RSA PRIVATE KEY-----...`

### ✅ Database
- [ ] `RDS_DATABASE_URL` = `postgresql://sentiment_user:PASSWORD@...`

---

## 🔍 How to Find Your Values

### Find Your .pem Files

```bash
# List all .pem files in common locations
find ~ -name "*.pem" -type f 2>/dev/null

# Common locations:
# ~/Downloads/
# ~/.ssh/
# ~/aws-keys/
```

### Get EC2 IPs from AWS Console

```bash
# Frontend (public IP):
1. Go to EC2 Console → Instances
2. Select your frontend instance
3. Copy "Public IPv4 address"

# Backend (private IP):
1. Select your backend instance  
2. Copy "Private IPv4 addresses"
```

### Get RDS Connection String

```bash
# From your current working setup:
echo $DATABASE_URL

# Or construct it:
postgresql://sentiment_user:<PASSWORD>@sentiment-db.cv2iq0sos7cc.ap-south-2.rds.amazonaws.com:5432/sentiment_db
```

---

## 🧪 Test Your Setup Locally

Before adding to GitHub, verify your keys work:

```bash
# Test frontend SSH
ssh -i ~/path/to/key.pem ubuntu@18.61.98.223 "echo 'Frontend OK'"

# Test backend SSH through frontend (jump host)
ssh -i ~/path/to/key.pem -J ubuntu@18.61.98.223 ubuntu@10.0.132.177 "echo 'Backend OK'"

# Test Docker Hub login
docker login -u sairam030

# Test DagsHub token
curl -H "Authorization: token YOUR_DAGSHUB_TOKEN" \
  https://dagshub.com/api/v1/user
```

---

## ⚠️ Common Issues

### Issue: "Permission denied (publickey)"
**Solution**: 
- Verify you copied the **entire** .pem file including BEGIN/END lines
- Check the username is correct (`ubuntu` vs `ec2-user`)
- Ensure the EC2 security group allows SSH from GitHub Actions IPs

### Issue: "Connection refused" to backend
**Solution**: 
- Backend is private, must use frontend as jump host
- Verify backend security group allows SSH from frontend EC2
- Check VPC routing tables

### Issue: "Invalid database URL"
**Solution**:
- Include password in connection string
- Escape special characters in password (use URL encoding)
- Verify RDS security group allows connections from backend EC2

### Issue: "Cannot pull from Docker Hub"
**Solution**:
- Regenerate Docker Hub token
- Ensure token has read/write permissions
- Verify username is `sairam030` in workflow

---

## 🚀 After Adding All Secrets

1. **Verify secrets are added**: Settings → Secrets and variables → Actions (should see 12 secrets)

2. **Push to trigger pipeline**:
```bash
git add .
git commit -m "Configure CI/CD pipeline"
git push origin main
```

3. **Monitor execution**: Actions tab → Click on workflow run

4. **Setup manual approval**:
   - Settings → Environments → New environment
   - Name: `production-approval`
   - Enable "Required reviewers" → Add yourself
   - Save protection rules

---

## 📊 Secrets Summary

| Category | Secrets | Status |
|----------|---------|--------|
| MLflow/DagsHub | 2 | Required before first run |
| Docker Hub | 1 | Required before first run |
| Test Frontend | 3 | Required before first run |
| Test Backend | 3 | Required before first run |
| Prod Frontend | 3 | Required before prod deploy |
| Prod Backend | 3 | Required before prod deploy |
| Database | 1 | Required before first run |
| **TOTAL** | **16** | **12 for test, 16 for full** |

---

## 💡 Pro Tips

1. **Same key for test environments**: If your test frontend and backend use the same .pem file, both secrets will have identical values.

2. **Different keys for prod**: For better security, use different key pairs for production.

3. **Rotate secrets**: Update Docker Hub and DagsHub tokens every 90 days.

4. **Backup .pem files**: Store securely - if lost, you'll need to create new EC2 key pairs.

5. **Test connection first**: Always test SSH locally before adding to GitHub secrets.

---

## ✅ Ready to Deploy?

Once all secrets are configured:

```bash
# Train models locally
dvc repro

# Promote to production
python scripts/promote_model.py --latest --alias production

# Push to trigger CI/CD
git push origin main

# Watch it deploy! 🚀
```
