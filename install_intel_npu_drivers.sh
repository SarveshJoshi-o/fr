#!/bin/bash

set -x

# Check Ubuntu version
if command -v lsb_release &> /dev/null; then
    UBUNTU_VERSION=$(lsb_release -rs)
elif [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
else
    # Fallback method
    UBUNTU_VERSION=$(cat /etc/issue | grep -o '[0-9]\+\.[0-9]\+' | head -1)
fi

echo "Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "24.04" ]]; then
    echo "Unsupported Ubuntu version: $UBUNTU_VERSION"
    echo "Falling back to Ubuntu 22.04 packages..."
    UBUNTU_VERSION="22.04"
fi

echo "Removing old packages..."
# dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu || true

# Define package URLs
if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    echo "Downloading packages for Ubuntu 22.04..."
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-driver-compiler-npu_1.13.0.20250131-13074932693_ubuntu22.04_amd64.deb"
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-fw-npu_1.13.0.20250131-13074932693_ubuntu22.04_amd64.deb"
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-level-zero-npu_1.13.0.20250131-13074932693_ubuntu22.04_amd64.deb"
    
    # Check if Level Zero is installed
    echo "Level Zero is missing. Downloading and installing for Ubuntu 22.04..."
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v1.18.5/level-zero_1.18.5+u22.04_amd64.deb"
    dpkg -i level-zero*.deb
    
elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    echo "Downloading packages for Ubuntu 24.04..."
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-driver-compiler-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb"
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-fw-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb"
    wget -q "https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-level-zero-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb"
    
    # Check if Level Zero is installed
    dpkg -l | grep level-zero || {
        echo "Level Zero is missing. Downloading and installing for Ubuntu 24.04..."
        wget -q "https://github.com/oneapi-src/level-zero/releases/download/v1.18.5/level-zero_1.18.5+u24.04_amd64.deb"
        dpkg -i level-zero*.deb
    }
fi

echo "Installing dependency libtbb12..."
apt update && apt install -y libtbb12

echo "Installing Intel NPU drivers..."
dpkg -i *.deb

echo "Intel NPU driver installation completed!"
