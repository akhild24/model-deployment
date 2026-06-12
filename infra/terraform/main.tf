# Root module: wires together VPC, IAM, and EKS modules for the ML Platform

# Networking layer — VPC, subnets, IGW, NAT, route tables
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
}

# IAM roles and policies — EKS cluster role, node role, OIDC provider, ALB controller role
module "iam" {
  source = "./modules/iam"

  project_name       = var.project_name
  environment        = var.environment
  eks_oidc_issuer_url = module.eks.oidc_issuer_url
}

# EKS cluster and managed node group — runs in private subnets with SPOT instances
module "eks" {
  source = "./modules/eks"

  project_name       = var.project_name
  environment        = var.environment
  cluster_version    = var.eks_cluster_version
  cluster_role_arn   = module.iam.eks_cluster_role_arn
  node_role_arn      = module.iam.eks_node_role_arn
  public_subnet_ids  = module.vpc.public_subnet_ids
  private_subnet_ids = module.vpc.private_subnet_ids
  node_instance_types = var.node_instance_types
  node_desired_size  = var.node_desired_size
  node_min_size      = var.node_min_size
  node_max_size      = var.node_max_size
}
