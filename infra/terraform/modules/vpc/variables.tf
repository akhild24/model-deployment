# Variables for the VPC module

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

# CIDR block for the VPC address space
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}
