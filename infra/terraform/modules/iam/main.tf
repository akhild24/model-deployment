# IAM module: creates roles, policies, and OIDC provider for EKS and ALB controller

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  # Strip the https:// prefix from the OIDC issuer URL for IAM trust policies
  oidc_issuer = replace(var.eks_oidc_issuer_url, "https://", "")
}

# Current AWS account ID for constructing ARNs
data "aws_caller_identity" "current" {}

# Current AWS partition (aws, aws-cn, aws-us-gov)
data "aws_partition" "current" {}

# ============================================================
# EKS CLUSTER ROLE
# ============================================================

# IAM role that the EKS control plane assumes to manage AWS resources
resource "aws_iam_role" "eks_cluster" {
  name = "${local.name_prefix}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "${local.name_prefix}-eks-cluster-role"
    Environment = var.environment
  }
}

# Attach the managed EKS cluster policy to the cluster role
resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# ============================================================
# EKS NODE GROUP ROLE
# ============================================================

# IAM role that EC2 worker nodes assume to join the EKS cluster
resource "aws_iam_role" "eks_node" {
  name = "${local.name_prefix}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "${local.name_prefix}-eks-node-role"
    Environment = var.environment
  }
}

# Attach the worker node policy — allows nodes to connect to EKS
resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node.name
}

# Attach the CNI policy — allows nodes to manage pod networking
resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node.name
}

# Attach the ECR read-only policy — allows nodes to pull container images
resource "aws_iam_role_policy_attachment" "eks_ecr_policy" {
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node.name
}

# ============================================================
# OIDC PROVIDER (for IAM Roles for Service Accounts - IRSA)
# ============================================================

# Fetch the TLS certificate for the EKS OIDC issuer to establish trust
data "tls_certificate" "eks" {
  url = var.eks_oidc_issuer_url
}

# OIDC identity provider — enables Kubernetes service accounts to assume IAM roles
resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = var.eks_oidc_issuer_url

  tags = {
    Name        = "${local.name_prefix}-eks-oidc"
    Environment = var.environment
  }
}

# ============================================================
# ALB CONTROLLER ROLE (using IRSA)
# ============================================================

# IAM role for the AWS Load Balancer Controller, scoped to its service account via IRSA
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
  policy      = file("${path.module}/alb-controller-policy.json")

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
