# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENVINO_VERSION=2024.6.0

# Install system dependencies including Intel GPU support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    pkg-config \
    lsb-release \
    # Intel GPU support
    intel-media-va-driver-non-free \
    libva-dev \
    libva-drm2 \
    libva-x11-2 \
    vainfo \
    # OpenCL support (Intel GPU drivers installed separately)
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# Install Intel GPU drivers first
COPY install_intel_gpu_drivers.sh /tmp/install_intel_gpu_drivers.sh
RUN chmod +x /tmp/install_intel_gpu_drivers.sh && \
    /tmp/install_intel_gpu_drivers.sh && \
    rm -rf /var/lib/apt/lists/*

# Install Intel NPU drivers
COPY install_intel_npu_drivers.sh /tmp/install_intel_npu_drivers.sh
RUN chmod +x /tmp/install_intel_npu_drivers.sh && \
    /tmp/install_intel_npu_drivers.sh && \
    rm -rf /var/lib/apt/lists/*

# Install Intel OpenVINO using modern GPG key method
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://apt.repos.intel.com/openvino/2024/gpgkey | gpg --dearmor -o /etc/apt/keyrings/openvino-2024.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/openvino-2024.gpg] https://apt.repos.intel.com/openvino/2024 ubuntu22 main" > /etc/apt/sources.list.d/intel-openvino-2024.list && \
    apt-get update && \
    apt-get install -y intel-openvino-runtime-ubuntu22-${OPENVINO_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# Set up OpenVINO environment
RUN echo "source /opt/intel/openvino_2024/setupvars.sh" >> /root/.bashrc

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install additional GPU/NPU dependencies
RUN pip3 install --no-cache-dir \
    openvino[extras] \
    openvino-dev[extras]

# Create working directory
WORKDIR /app

# Copy only the current directory contents (Code folder)
COPY . /app/

# Create necessary directories (these will be overridden by volume mounts)
RUN mkdir -p /app/videos/input /app/videos/output /app/database/Photos

# Set up permissions
RUN chmod +x /app/main.py

# Create a test script for GPU/NPU detection
RUN echo '#!/bin/bash\n\
echo "=== System Information ==="\n\
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME)"\n\
echo "Python: $(python3 --version)"\n\
echo "OpenVINO: $(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "Not available")"\n\
echo ""\n\
echo "=== Available Devices ==="\n\
python3 -c "\n\
import openvino as ov\n\
core = ov.Core()\n\
available_devices = core.available_devices\n\
print(f\"Available devices: {available_devices}\")\n\
for device in available_devices:\n\
    try:\n\
        device_name = core.get_property(device, \"FULL_DEVICE_NAME\")\n\
        print(f\"  {device}: {device_name}\")\n\
    except Exception as e:\n\
        print(f\"  {device}: Error getting device info - {e}\")\n\
"\n\
echo ""\n\
echo "=== Intel GPU Information ==="\n\
if [ -d "/dev/dri" ]; then\n\
    echo "DRI devices available:"\n\
    ls -la /dev/dri/\n\
    echo ""\n\
    if command -v vainfo &> /dev/null; then\n\
        echo "VA-API Info:"\n\
        vainfo\n\
    else\n\
        echo "vainfo not available"\n\
    fi\n\
    echo ""\n\
    if command -v clinfo &> /dev/null; then\n\
        echo "OpenCL Info:"\n\
        clinfo | head -20\n\
    else\n\
        echo "clinfo not available"\n\
    fi\n\
else\n\
    echo "Intel GPU not accessible (/dev/dri not found)"\n\
    echo "Run with: --device /dev/dri:/dev/dri --privileged"\n\
fi\n\
echo ""\n\
echo "=== NVIDIA GPU Information ==="\n\
if command -v nvidia-smi &> /dev/null; then\n\
    nvidia-smi\n\
else\n\
    echo "NVIDIA GPU not detected or nvidia-smi not available"\n\
fi\n\
echo ""\n\
echo "=== NPU Information ==="\n\
if command -v lspci &> /dev/null; then\n\
    lspci | grep -i "neural\|npu\|ai\|accelerator" || echo "No NPU devices found via lspci"\n\
else\n\
    echo "lspci not available"\n\
fi\n\
echo ""\n\
echo "=== Running Intel GPU Test ==="\n\
python3 /app/test_intel_gpu.py\n\
echo ""\n\
echo "=== Running Intel NPU Test ==="\n\
python3 /app/test_intel_npu.py\n\
echo ""\n\
echo "=== Running OpenCV and OpenCL Test ==="\n\
python3 /app/test_opencv_opencl.py\n\
' > /app/test_devices.sh && chmod +x /app/test_devices.sh

# Expose port (if needed for web interface)
EXPOSE 8000

# Default command
CMD ["/app/test_devices.sh"]
