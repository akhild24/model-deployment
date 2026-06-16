# Root-level variables for the ML Platform infrastructure

# AWS region to deploy all resources into
variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

# Project name used as a prefix for all resource names
variable "project_name" {
  description = "Name of the project, used as prefix for resource naming"
  type        = string
  default     = "ml-platform"
}

# Deployment environment (dev, staging, prod)
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

# CIDR block for the VPC
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Kubernetes version for the EKS cluster
variable "eks_cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.30"
}

# EC2 instance types for EKS worker nodes
variable "node_instance_types" {
  description = "List of EC2 instance types for EKS node group"
  type        = list(string)
  default     = ["t3.medium"]
}

# Desired number of worker nodes
variable "node_desired_size" {
  description = "Desired number of worker nodes in the node group"
  type        = number
  default     = 2
}

# Minimum number of worker nodes
variable "node_min_size" {
  description = "Minimum number of worker nodes in the node group"
  type        = number
  default     = 1
}

# Maximum number of worker nodes
variable "node_max_size" {
  description = "Maximum number of worker nodes in the node group"
  type        = number
  default     = 4
}
