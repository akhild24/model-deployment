# Outputs for the IAM module

# ARN of the IAM role for the EKS control plane
output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster.arn
}

# ARN of the IAM role for EKS worker nodes
output "eks_node_role_arn" {
  description = "ARN of the EKS node group IAM role"
  value       = aws_iam_role.eks_node.arn
}

# ARN of the IAM role for the AWS Load Balancer Controller
output "alb_controller_role_arn" {
  description = "ARN of the ALB controller IAM role"
  value       = aws_iam_role.alb_controller.arn
}

# ARN of the OIDC identity provider for IRSA
output "oidc_provider_arn" {
  description = "ARN of the OIDC identity provider"
  value       = aws_iam_openid_connect_provider.eks.arn
}
