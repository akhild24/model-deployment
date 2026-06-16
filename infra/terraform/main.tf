# Root module: wires together VPC, IAM, and EKS modules for the ML Platform

# Networking layer — VPC, subnets, IGW, NAT, route tables
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
}

# IAM roles — EKS cluster role and node role (no EKS dependency)
module "iam" {
  source = "./modules/iam"

  project_name = var.project_name
  environment  = var.environment
}

# EKS cluster and managed node group — runs in private subnets with SPOT instances
module "eks" {
  source = "./modules/eks"

  project_name        = var.project_name
  environment         = var.environment
  cluster_version     = var.eks_cluster_version
  cluster_role_arn    = module.iam.eks_cluster_role_arn
  node_role_arn       = module.iam.eks_node_role_arn
  public_subnet_ids   = module.vpc.public_subnet_ids
  private_subnet_ids  = module.vpc.private_subnet_ids
  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_min_size       = var.node_min_size
  node_max_size       = var.node_max_size
}

# ============================================================
# OIDC PROVIDER (for IAM Roles for Service Accounts - IRSA)
# Placed at root level to avoid circular dependency with EKS
# ============================================================

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  oidc_issuer = replace(module.eks.oidc_issuer_url, "https://", "")
}

# Fetch the TLS certificate for the EKS OIDC issuer
data "tls_certificate" "eks" {
  url = module.eks.oidc_issuer_url
}

# OIDC identity provider — enables K8s service accounts to assume IAM roles
resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = module.eks.oidc_issuer_url

  tags = {
    Name        = "${local.name_prefix}-eks-oidc"
    Environment = var.environment
  }
}

# ============================================================
# ALB CONTROLLER ROLE (using IRSA)
# ============================================================

# IAM role for the AWS Load Balancer Controller
resource "aws_iam_role" "alb_controller" {
  name = "${local.name_prefix}-alb-controller-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.eks.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
            "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:aws-load-balancer-controller"
          }
        }
      }
    ]
  })

  tags = {
    Name        = "${local.name_prefix}-alb-controller-role"
    Environment = var.environment
  }
}

# IAM policy for ALB controller — loaded from the official AWS policy JSON
resource "aws_iam_policy" "alb_controller" {
  name        = "${local.name_prefix}-alb-controller-policy"
  description = "IAM policy for the AWS Load Balancer Controller"
  policy      = file("${path.module}/modules/iam/alb-controller-policy.json")

  tags = {
    Name        = "${local.name_prefix}-alb-controller-policy"
    Environment = var.environment
  }
}

# Attach the ALB controller policy to the ALB controller role
resource "aws_iam_role_policy_attachment" "alb_controller" {
  policy_arn = aws_iam_policy.alb_controller.arn
  role       = aws_iam_role.alb_controller.name
}
