#!/bin/bash

# Script to install Intel GPU compute drivers on Ubuntu 22.04 LTS

set -x

# Install Intel graphics GPG public key
echo "Installing Intel graphics GPG public key..."
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Configure Intel GPU repository
echo "Configuring the Intel GPU package repository..."
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Update package repository metadata
echo "Updating package repositories..."
apt update

# Install essential Intel GPU compute-related packages
echo "Installing Intel GPU compute-related packages..."
apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo

# Additional package installations (optional, depending on your use case)
echo "Checking for additional packages based on your needs..."

# Check if PyTorch is needed (uncomment if you need it)
# echo "Installing PyTorch dependencies..."
# apt-get install -y libze-dev intel-ocloc

# Check if hardware ray tracing is needed (uncomment if you need it)
# echo "Installing hardware ray tracing support..."
# apt-get install -y intel-level-zero-gpu-raytracing

# Verifying the installation using clinfo
echo "Verifying the installation..."
clinfo | grep "Device Name"

# Output completion message
echo "Intel GPU compute driver installation completed!"
