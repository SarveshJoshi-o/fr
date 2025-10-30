# Docker Setup Guide for Face Recognition with GPU/NPU Support

This guide explains how to create and run a Docker container for your face recognition application with GPU and NPU acceleration support.

## Prerequisites

### 1. Docker Installation
Make sure Docker is installed and you have permission to run Docker commands:

```bash
# Check Docker installation
docker --version

# If you get permission denied, add your user to docker group:
sudo usermod -aG docker $USER
# Then logout and login again, or run:
newgrp docker
```

### 2. NVIDIA Docker Support (for GPU acceleration)
If you have NVIDIA GPU and want GPU acceleration:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Intel NPU Support
For Intel NPU support, ensure you have the necessary drivers installed on your host system.

## Building and Running

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and test the container
docker-compose build

# Run device detection tests
docker-compose run --rm face-recognition

# Run the comprehensive test suite
docker-compose run --rm face-recognition python3 /app/test_gpu_npu.py

# Run your face recognition application
docker-compose run --rm face-recognition python3 /app/main.py
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t ov_fr_test .

# Run with GPU support (if NVIDIA GPU available)
docker run --rm --gpus all -v $(pwd)/videos:/app/videos -v $(pwd)/database:/app/database -v $(pwd)/buffalo_l:/app/buffalo_l ov_fr_test

# Run with CPU only
docker run --rm -v $(pwd)/videos:/app/videos -v $(pwd)/database:/app/database -v $(pwd)/buffalo_l:/app/buffalo_l ov_fr_test
```

### Option 3: Interactive Container

```bash
# Run interactive container for debugging
docker run -it --rm --gpus all \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/database:/app/database \
  -v $(pwd)/buffalo_l:/app/buffalo_l \
  -v $(pwd)/final_embeddings_adaface_ov_fp16.jsonl:/app/final_embeddings_adaface_ov_fp16.jsonl \
  ov_fr_test /bin/bash
```

## Testing GPU/NPU Access

### 1. Run the built-in test script:
```bash
docker-compose run --rm face-recognition python3 /app/test_gpu_npu.py
```

### 2. Manual testing inside container:
```bash
# Enter the container
docker-compose run --rm face-recognition /bin/bash

# Inside the container, run:
python3 /app/test_gpu_npu.py
```

## Expected Output

When running successfully, you should see:

1. **System Information**: OS, Python version, memory
2. **OpenVINO Devices**: Available devices (CPU, GPU, NPU)
3. **NVIDIA GPU**: GPU information if available
4. **Intel NPU**: NPU device information if available
5. **Model Loading**: Success/failure of loading face recognition models

## Troubleshooting

### Common Issues:

1. **Permission Denied**: Add user to docker group and restart session
2. **NVIDIA GPU not detected**: Install NVIDIA Container Toolkit
3. **NPU not detected**: Ensure Intel NPU drivers are installed on host
4. **Model files not found**: Check that buffalo_l directory is properly mounted

### Debug Commands:

```bash
# Check Docker daemon status
sudo systemctl status docker

# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check available devices in container
docker run --rm ov_fr_test python3 -c "import openvino as ov; print(ov.Core().available_devices)"
```

## Performance Optimization

### For GPU Acceleration:
- Ensure your models are optimized for GPU inference
- Use appropriate batch sizes
- Monitor GPU memory usage

### For NPU Acceleration:
- Use Intel OpenVINO NPU plugin
- Optimize models for NPU inference
- Check NPU utilization

## File Structure

```
/home/sr/ov_fr/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── test_gpu_npu.py         # GPU/NPU testing script
├── build_and_run.sh        # Build and run script
├── .dockerignore           # Files to exclude from Docker build
├── main.py                 # Your face recognition application
├── requirements.txt        # Python dependencies
├── buffalo_l/              # Model files (mounted as volume)
├── videos/                 # Video files (mounted as volume)
└── database/               # Database files (mounted as volume)
```

## Next Steps

1. **Test the container**: Run the test script to verify GPU/NPU access
2. **Run your application**: Execute main.py with your video files
3. **Monitor performance**: Check GPU/NPU utilization during inference
4. **Optimize models**: Ensure models are optimized for your target hardware

## Support

If you encounter issues:
1. Check the Docker logs: `docker-compose logs face-recognition`
2. Run the test script: `python3 /app/test_gpu_npu.py`
3. Verify hardware access: Check device availability in the container
