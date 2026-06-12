#!/bin/bash
set -e
export PATH="$HOME/bin:$PATH"

echo "=== Downloading ALB Controller IAM Policy ==="
curl -o modules/iam/alb-controller-policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.7.1/docs/install/iam_policy.json

echo "=== Terraform Init ==="
terraform init

echo "=== Creating Workspaces ==="
terraform workspace new dev 2>/dev/null || echo "dev workspace already exists"
terraform workspace new prod 2>/dev/null || echo "prod workspace already exists"
terraform workspace select dev

echo "=== Running Plan ==="
terraform plan

echo "Done. Review the plan above, then run: terraform apply"
