# Variables for the EKS module

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

# Kubernetes version for the EKS cluster
variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
}

# IAM role ARN for the EKS cluster control plane
variable "cluster_role_arn" {
  description = "ARN of the IAM role for the EKS cluster"
  type        = string
}

# IAM role ARN for the EKS worker nodes
variable "node_role_arn" {
  description = "ARN of the IAM role for the EKS node group"
  type        = string
}

# Public subnet IDs for the EKS cluster VPC config
variable "public_subnet_ids" {
  description = "IDs of the public subnets"
  type        = list(string)
}

# Private subnet IDs where worker nodes will be launched
variable "private_subnet_ids" {
  description = "IDs of the private subnets"
  type        = list(string)
}

# EC2 instance types for the node group
variable "node_instance_types" {
  description = "List of EC2 instance types for the node group"
  type        = list(string)
}

# Desired number of worker nodes
variable "node_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
}

# Minimum number of worker nodes
variable "node_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
}

# Maximum number of worker nodes
variable "node_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
}
