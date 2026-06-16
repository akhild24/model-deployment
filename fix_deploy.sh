#!/bin/bash
set -e

echo "========================================="
echo "  ML Platform - Fix & Deploy Script"
echo "========================================="
echo ""

cd ~/model-deployment

# -----------------------------------------------
# Step 1: Create ECR Repository
# -----------------------------------------------
echo "[1/6] Creating ECR repository..."
aws ecr create-repository --repository-name ml-platform --region us-east-1 2>/dev/null \
  && echo "  ✅ Repository created" \
  || echo "  ℹ️  Repository already exists"

# -----------------------------------------------
# Step 2: Login to ECR
# -----------------------------------------------
echo ""
echo "[2/6] Logging into ECR..."
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 492094933467.dkr.ecr.us-east-1.amazonaws.com
echo "  ✅ ECR login successful"

# -----------------------------------------------
# Step 3: Build Docker image
# -----------------------------------------------
echo ""
echo "[3/6] Building Docker image (this may take a few minutes)..."
docker build -t ml-platform .
echo "  ✅ Docker build complete"

# -----------------------------------------------
# Step 4: Tag and push to ECR
# -----------------------------------------------
echo ""
echo "[4/6] Pushing image to ECR..."
docker tag ml-platform:latest 492094933467.dkr.ecr.us-east-1.amazonaws.com/ml-platform:latest
docker push 492094933467.dkr.ecr.us-east-1.amazonaws.com/ml-platform:latest
echo "  ✅ Image pushed to ECR"

# -----------------------------------------------
# Step 5: Restart deployment to pull new image
# -----------------------------------------------
echo ""
echo "[5/6] Restarting deployment to pull the image..."
kubectl rollout restart deployment ml-serving -n ml-platform
echo "  ✅ Rollout restart triggered"

# Wait for pods to come up
echo "  ⏳ Waiting for pods to be ready (up to 120s)..."
kubectl rollout status deployment/ml-serving -n ml-platform --timeout=120s || true

# -----------------------------------------------
# Step 6: Check ALB controller
# -----------------------------------------------
echo ""
echo "[6/6] Checking ALB Controller status..."
echo ""
echo "--- ALB Controller Pods ---"
kubectl get pods -n kube-system | grep aws-load-balancer || echo "  ⚠️  ALB Controller not found"
echo ""
echo "--- ALB Controller Logs (last 15 lines) ---"
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller --tail=15 2>/dev/null || echo "  ⚠️  Could not fetch logs"

# -----------------------------------------------
# Final Status
# -----------------------------------------------
echo ""
echo "========================================="
echo "  Final Status"
echo "========================================="
echo ""
echo "--- All Resources in ml-platform ---"
kubectl get all -n ml-platform
echo ""
echo "--- Ingress ---"
kubectl get ingress -n ml-platform
echo ""
echo "--- Node Status ---"
kubectl get nodes
echo ""
echo "========================================="
echo "  Done! Check output above for status."
echo "========================================="
