# Variables for the IAM module

# Project name used in resource naming
variable "project_name" {
  description = "Name of the project"
  type        = string
}

# Deployment environment used in resource naming and tags
variable "environment" {
  description = "Deployment environment"
  type        = string
}

# OIDC issuer URL from the EKS cluster — used to create the OIDC provider and IRSA trust
variable "eks_oidc_issuer_url" {
  description = "OIDC issuer URL from the EKS cluster"
  type        = string
}
