# Root-level outputs for the ML Platform infrastructure

# Name of the EKS cluster
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

# API endpoint for the EKS cluster
output "cluster_endpoint" {
  description = "Endpoint URL for the EKS cluster API server"
  value       = module.eks.cluster_endpoint
}

# VPC ID where all resources are deployed
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

# IAM role ARN for the ALB Ingress Controller
output "alb_role_arn" {
  description = "IAM role ARN for the AWS Load Balancer Controller"
  value       = module.iam.alb_controller_role_arn
}
