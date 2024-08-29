terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "ssh_pub_key_file" {
  type    = string
  default = "~/.ssh/id_rsa.pub"
}


variable "region" {
  type    = string
  default = "us-west-1"
}

variable "base_name" {
  type    = string
  default = "chunkmydocs-vpc"
}

###############################################################
# VPC configuration
###############################################################
# Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
provider "aws" {
  region = var.region
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name   = "${var.base_name}-vpc"

  enable_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  cidr                                   = "10.0.0.0/16"
  azs                                    = ["us-west-1a", "us-west-1c"]
  private_subnets                        = ["10.0.0.0/24", "10.0.32.0/19", "10.0.64.0/19"]
  public_subnets                         = ["10.0.101.0/24", "10.0.102.0/24", "10.0.160.0/19"]
  create_database_subnet_group           = true
  create_database_subnet_route_table     = true
  create_database_internet_gateway_route = true


  private_subnet_tags = {
    "kubernetes.io/cluster/${var.base_name}" : "shared"
    "kubernetes.io/role/internal-elb" = "1"
  }

  public_subnet_tags = {
    "kubernetes.io/role/elb" : 1
  }
}

###############################################################
# VM configuration
###############################################################


resource "aws_instance" "vm1" {
  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20240501
  # https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-22-04/
  ami = "ami-0b5b200dc06507fcb"

  instance_type = "g5.xlarge"
  user_data = base64encode(templatefile("./vm_init.yaml", {
    ssh_key : file(var.ssh_pub_key_file)
  }))
  user_data_replace_on_change = true

  subnet_id                   = module.vpc.public_subnets[0]
  vpc_security_group_ids      = [aws_security_group.vm_security_group.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 200 # In GB
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.base_name}-vm"
  }
}

resource "aws_security_group" "vm_security_group" {
  name   = "vm_security_group"
  vpc_id = module.vpc.vpc_id

  # SSH access from the VPC
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 6000
    to_port     = 6000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 7000
    to_port     = 7000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

output "vm_public_ip" {
  value = aws_instance.vm1.public_ip
}
