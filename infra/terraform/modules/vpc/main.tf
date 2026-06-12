# VPC module: creates the networking foundation for the ML Platform

# Fetch available AZs in the current region
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)
}

# VPC with DNS support enabled for EKS service discovery
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "${local.name_prefix}-vpc"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Public subnet in AZ-0 — hosts NAT Gateway and load balancers
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                               = "${local.name_prefix}-public-${local.azs[count.index]}"
    Environment                                        = var.environment
    "kubernetes.io/cluster/${local.name_prefix}"        = "owned"
    "kubernetes.io/role/elb"                            = "1"
  }
}

# Private subnet — hosts EKS worker nodes (no direct internet access)
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = local.azs[count.index]

  tags = {
    Name                                               = "${local.name_prefix}-private-${local.azs[count.index]}"
    Environment                                        = var.environment
    "kubernetes.io/cluster/${local.name_prefix}"        = "owned"
    "kubernetes.io/role/internal-elb"                   = "1"
  }
}

# Internet Gateway — provides internet access for public subnets
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${local.name_prefix}-igw"
    Environment = var.environment
  }
}

# Elastic IP for the NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name        = "${local.name_prefix}-nat-eip"
    Environment = var.environment
  }
}

# NAT Gateway — allows private subnet resources to reach the internet
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name        = "${local.name_prefix}-nat"
    Environment = var.environment
  }

  depends_on = [aws_internet_gateway.main]
}

# Public route table — routes internet traffic through the IGW
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${local.name_prefix}-public-rt"
    Environment = var.environment
  }
}

# Private route table — routes internet traffic through the NAT Gateway
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name        = "${local.name_prefix}-private-rt"
    Environment = var.environment
  }
}

# Associate public subnets with the public route table
resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Associate private subnets with the private route table
resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
