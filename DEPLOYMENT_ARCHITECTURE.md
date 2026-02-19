# Deployment Architecture Options

## Option 1: Single EC2 Per Environment (RECOMMENDED)
**Simpler, cheaper, sufficient for most use cases**

```
┌─────────────────────────────────────┐
│   TEST EC2 (18.61.98.223)           │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Frontend Container (Port 80)│  │
│  │  - Nginx                      │  │
│  │  - Proxies /api/ to backend  │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ Backend Container (Port 5000)│  │
│  │  - Flask + Gunicorn          │  │
│  │  - ML Models                 │  │
│  └──────────────────────────────┘  │
│                                     │
│  Docker Compose manages both        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   PROD EC2 (Different IP)           │
│                                     │
│  Same structure as TEST             │
│  Both containers on one instance    │
└─────────────────────────────────────┘
```

### GitHub Secrets for Option 1:

```
TEST_EC2_HOST=18.61.98.223                    # Frontend public IP
TEST_EC2_USER=ubuntu                          # or ec2-user
TEST_EC2_SSH_KEY=<your-key.pem-content>       # Same key, one instance

PROD_EC2_HOST=<prod-public-ip>                # Different EC2
PROD_EC2_USER=ubuntu
PROD_EC2_SSH_KEY=<prod-key.pem-content>       # Could be same or different key
PROD_BACKEND_HOST=<prod-private-ip>           # Private IP of SAME instance
```

### Pros:
✅ Cheaper (1 EC2 per environment = 2 total)
✅ Simpler to manage
✅ Sufficient for moderate traffic
✅ Docker Compose handles networking

### Cons:
❌ Can't scale frontend/backend independently
❌ Both services share same resources

---

## Option 2: Separate EC2s Per Service (SCALABLE)
**More complex, more expensive, better for high traffic**

```
TEST ENVIRONMENT:
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ TEST Frontend EC2            │     │ TEST Backend EC2             │
│ (18.61.98.223 - Public)      │────▶│ (10.0.132.177 - Private)     │
│                              │     │                              │
│ - Nginx on port 80           │     │ - Flask + Gunicorn           │
│ - Proxies to backend         │     │ - ML Models                  │
└──────────────────────────────┘     └──────────────────────────────┘

PROD ENVIRONMENT:
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ PROD Frontend EC2            │     │ PROD Backend EC2             │
│ (New IP - Public)            │────▶│ (New IP - Private)           │
│                              │     │                              │
│ - Nginx on port 80           │     │ - Flask + Gunicorn           │
│ - Proxies to backend         │     │ - ML Models                  │
└──────────────────────────────┘     └──────────────────────────────┘
```

### GitHub Secrets for Option 2:

```
# Test Frontend
TEST_FRONTEND_HOST=18.61.98.223
TEST_FRONTEND_USER=ubuntu
TEST_FRONTEND_SSH_KEY=<frontend-key.pem>

# Test Backend
TEST_BACKEND_HOST=10.0.132.177              # Private IP
TEST_BACKEND_USER=ubuntu
TEST_BACKEND_SSH_KEY=<backend-key.pem>

# Prod Frontend
PROD_FRONTEND_HOST=<new-prod-frontend-ip>
PROD_FRONTEND_USER=ubuntu
PROD_FRONTEND_SSH_KEY=<prod-frontend-key.pem>

# Prod Backend
PROD_BACKEND_HOST=<new-prod-backend-private-ip>
PROD_BACKEND_USER=ubuntu
PROD_BACKEND_SSH_KEY=<prod-backend-key.pem>
```

### Pros:
✅ Scale frontend/backend independently
✅ Backend in private subnet (more secure)
✅ Can use load balancers
✅ Better for high traffic

### Cons:
❌ 4 EC2 instances = 2x cost
❌ More complex deployment
❌ Need to deploy to 2 instances per environment

---

## What You Currently Have

Looking at your setup:
- **Test**: 2 EC2s (frontend public, backend private)
- **Both running docker-compose** → Actually running both containers on frontend EC2

You're in a **hybrid state**. You have 2 EC2s but using docker-compose which runs both on one.

---

## Recommendation

### For Your Use Case: **Option 1** (Single EC2)

Why:
- Your docker-compose.yml already runs both containers together
- You're not at scale requiring separation
- Saves money (2 EC2s total vs 4)
- Simpler CI/CD pipeline
- You can migrate to Option 2 later if needed

### Migration Steps:

**Current State:**
- Frontend EC2: 18.61.98.223 (running docker-compose with BOTH containers)
- Backend EC2: 10.0.132.177 (not actually used? or running separately?)

**Clarify Your Setup:**

Are you currently running:
1. Docker Compose on frontend EC2 (both containers together)?
2. OR separate deployments on each EC2?

Run this to check:
```bash
# On frontend EC2 (18.61.98.223)
ssh -i your-key.pem ubuntu@18.61.98.223 "docker ps"

# On backend EC2 (10.0.132.177) 
ssh -i your-key.pem ubuntu@10.0.132.177 "docker ps"
```

Tell me what you see, and I'll adjust the CI/CD pipeline accordingly!
