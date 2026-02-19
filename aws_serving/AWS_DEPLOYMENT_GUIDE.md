# AWS Deployment Guide with RDS PostgreSQL

## Architecture Overview

```
Internet → Frontend EC2 (Public) → Backend EC2 (Private) → RDS PostgreSQL (Private)
```

---

## Step 1: Create AWS RDS PostgreSQL Database

### 1.1 Create RDS Instance

1. Go to **AWS Console → RDS → Create database**
2. Choose:
   - Engine: **PostgreSQL 15**
   - Template: **Free tier** (for testing) or **Production** (for prod)
   - DB instance identifier: `sentiment-db`
   - Master username: `sentiment_user`
   - Master password: `sentiment_pass` (choose a strong password)
   - DB instance class: `db.t3.micro` (free tier) or `db.t3.small`
   - Storage: 20 GB SSD
   - VPC: **Your VPC** (same as your EC2 instances)
   - Subnet group: **Private subnets only**
   - Public access: **No**
   - VPC security group: Create new `sentiment-rds-sg`
   - Database name: `sentiment_db`

3. Click **Create database**

### 1.2 Configure RDS Security Group

Add inbound rule to `sentiment-rds-sg`:
- **Type:** PostgreSQL
- **Port:** 5432
- **Source:** Security group of Backend EC2 (e.g., `sg-backend-xxx`)
- **Description:** Allow backend to connect

### 1.3 Note the RDS Endpoint

After creation, note the endpoint:
```
sentiment-db.xxxxx.us-east-1.rds.amazonaws.com
```

---

## Step 2: Build and Push Updated Docker Images

Run locally:

```bash
cd /home/ram/sentiment_mlops/aws_serving

# Build backend with database support
docker build -t sairam030/sentiment-backend:test -f app/Dockerfile app/
docker push sairam030/sentiment-backend:test

# Build frontend with environment variable support
docker build -t sairam030/sentiment-frontend:test -f frontend/Dockerfile frontend/
docker push sairam030/sentiment-frontend:test
```

---

## Step 3: Deploy Backend on Private EC2

### 3.1 SSH into Backend EC2

```bash
# Replace with your backend EC2 private IP (use bastion or SSM)
ssh -i "sshmlops.pem" ubuntu@<BACKEND_PRIVATE_IP>
```

### 3.2 Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
exit  # Log out and back in
```

### 3.3 Create docker-compose.yml

```bash
mkdir -p ~/sentiment_app && cd ~/sentiment_app

cat > docker-compose.yml << 'EOF'
version: "3.8"

services:
  backend:
    image: sairam030/sentiment-backend:test
    container_name: sentiment-backend-test
    ports:
      - "5000:5000"
    environment:
      # Replace with YOUR RDS endpoint
      - DATABASE_URL=postgresql://sentiment_user:YOUR_PASSWORD@sentiment-db.xxxxx.us-east-1.rds.amazonaws.com:5432/sentiment_db
      - MODEL_VERSION=v1.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
    restart: unless-stopped
EOF
```

**⚠️ Important:** Replace:
- `YOUR_PASSWORD` with your RDS password
- `sentiment-db.xxxxx.us-east-1.rds.amazonaws.com` with your RDS endpoint

### 3.4 Deploy Backend

```bash
docker compose pull
docker compose up -d
```

### 3.5 Verify Backend

```bash
docker ps
docker logs sentiment-backend-test

# Test health endpoint
curl http://localhost:5000/health

# Test monitoring endpoints
curl http://localhost:5000/monitoring/stats
curl http://localhost:5000/monitoring/drift
```

### 3.6 Note Backend Private IP

```bash
hostname -I
# Example: 10.0.132.177
```

---

## Step 4: Deploy Frontend on Public EC2

### 4.1 SSH into Frontend EC2

```bash
ssh -i "sshmlops.pem" ubuntu@18.61.235.160
```

### 4.2 Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
exit  # Log out and back in
```

### 4.3 Create docker-compose.yml

```bash
mkdir -p ~/sentiment_app && cd ~/sentiment_app

cat > docker-compose.yml << 'EOF'
version: "3.8"

services:
  frontend:
    image: sairam030/sentiment-frontend:test
    container_name: sentiment-frontend-test
    ports:
      - "80:80"
    environment:
      # Replace with your backend EC2 private IP
      - BACKEND_HOST=10.0.132.177
      - BACKEND_PORT=5000
    restart: unless-stopped
EOF
```

**⚠️ Important:** Replace `10.0.132.177` with your backend EC2's actual private IP.

### 4.4 Deploy Frontend

```bash
docker compose pull
docker compose up -d
```

### 4.5 Verify Frontend

```bash
docker ps
docker logs sentiment-frontend-test

# Test locally
curl http://localhost/api/health
```

---

## Step 5: Test the Full System

From your local machine:

```bash
# Test frontend access
curl http://18.61.235.160/api/health

# Test prediction
curl -X POST http://18.61.235.160/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'

# Check monitoring stats
curl http://18.61.235.160/api/monitoring/stats

# Check drift detection
curl http://18.61.235.160/api/monitoring/drift
```

---

## Step 6: Security Group Configuration

### Backend EC2 Security Group
**Inbound:**
- Port `5000` from Frontend EC2 security group
- Port `22` (SSH) from your IP or bastion

**Outbound:**
- Port `5432` to RDS security group
- Port `443` to internet (for Docker Hub)

### Frontend EC2 Security Group
**Inbound:**
- Port `80` from `0.0.0.0/0` (public HTTP)
- Port `443` from `0.0.0.0/0` (public HTTPS, if SSL)
- Port `22` (SSH) from your IP

**Outbound:**
- Port `5000` to Backend EC2 security group
- Port `443` to internet (for Docker Hub)

### RDS Security Group
**Inbound:**
- Port `5432` from Backend EC2 security group

---

## Step 7: Query the Database (Optional)

Connect to RDS from backend EC2:

```bash
# Install psql client on backend EC2
sudo apt install -y postgresql-client

# Connect to RDS
psql -h sentiment-db.xxxxx.us-east-1.rds.amazonaws.com \
     -U sentiment_user \
     -d sentiment_db

# Query predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;

# Count by label
SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction;

# Average confidence by day
SELECT DATE(timestamp), AVG(confidence) 
FROM predictions 
GROUP BY DATE(timestamp) 
ORDER BY DATE(timestamp) DESC;
```

---

## Monitoring Endpoints

Access via frontend URL:

- **Health Check:** `http://18.61.235.160/api/health`
- **Prediction Stats:** `http://18.61.235.160/api/monitoring/stats`
- **Drift Detection:** `http://18.61.235.160/api/monitoring/drift`

---

## Production Deployment

For production, use `:prod` tags:

```bash
# Tag and push prod images
docker tag sairam030/sentiment-backend:test sairam030/sentiment-backend:prod
docker tag sairam030/sentiment-frontend:test sairam030/sentiment-frontend:prod
docker push sairam030/sentiment-backend:prod
docker push sairam030/sentiment-frontend:prod

# Update docker-compose.yml to use :prod tags
# Then: docker compose pull && docker compose up -d
```

---

## Troubleshooting

### Backend can't connect to RDS
- Check RDS security group allows port 5432 from backend SG
- Verify DATABASE_URL is correct
- Check RDS is in same VPC as backend EC2

### Frontend can't reach backend
- Verify BACKEND_HOST is the backend's **private IP**
- Check backend security group allows port 5000 from frontend SG
- Test: `curl http://<backend-private-ip>:5000/health` from frontend EC2

### Database tables not created
- Backend logs will show: `[database] ✅ Database tables initialized`
- If not, manually create: Connect with psql and run schema from database.py

---

## Cost Estimation (AWS US-East-1)

- **RDS db.t3.micro (20GB):** ~$15-20/month
- **EC2 t2.micro × 2:** Free tier (12 months) or ~$17/month
- **Total:** ~$15-40/month

---

## Next Steps

1. Set up CloudWatch alarms for drift detection
2. Configure RDS automated backups
3. Add SSL certificate for HTTPS
4. Set up CI/CD pipeline for automated deployments
5. Implement request authentication/rate limiting
