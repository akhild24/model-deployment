# IAM module: creates core IAM roles and policies for EKS cluster and nodes
# OIDC provider and ALB controller are handled at root level (they depend on EKS outputs)

locals {
  name_prefix = "${var.project_name}-${var.environment}"
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
