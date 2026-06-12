# Outputs for the VPC module

# ID of the created VPC
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

# List of public subnet IDs (for load balancers and IGW access)
output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

# List of private subnet IDs (for EKS worker nodes)
output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}
