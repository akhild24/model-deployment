# EKS module: creates the Kubernetes cluster and managed node group

locals {
  name_prefix  = "${var.project_name}-${var.environment}"
  cluster_name = "${local.name_prefix}-cluster"
}

# EKS cluster — managed Kubernetes control plane
resource "aws_eks_cluster" "main" {
  name     = local.cluster_name
  version  = var.cluster_version
  role_arn = var.cluster_role_arn

  vpc_config {
    # Place the control plane ENIs in both public and private subnets
    subnet_ids              = concat(var.public_subnet_ids, var.private_subnet_ids)
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  tags = {
    Name        = local.cluster_name
    Environment = var.environment
    Project     = var.project_name
  }
}

# EKS managed node group — SPOT instances in private subnets for cost savings
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${local.name_prefix}-node-group"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = var.node_instance_types
  capacity_type   = "SPOT"

  scaling_config {
    desired_size = var.node_desired_size
    min_size     = var.node_min_size
    max_size     = var.node_max_size
  }

  tags = {
    Name        = "${local.name_prefix}-node-group"
    Environment = var.environment
    Project     = var.project_name
  }
}
