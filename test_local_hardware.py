#!/usr/bin/env python3
"""
Test script to check local hardware capabilities before Docker setup
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and return its output"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            print(result.stdout)
        else:
            print(f"✗ {description} - FAILED")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def test_system():
    """Test system information"""
    print("🔍 System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    print(f"Total Memory: {line.strip()}")
                    break
    except:
        print("Could not read memory info")

def test_docker():
    """Test Docker installation and permissions"""
    print("\n🔍 Docker Status:")
    
    # Check if Docker is installed
    docker_installed = run_command("docker --version", "Docker Installation")
    
    if docker_installed:
        # Check Docker daemon status
        run_command("docker info", "Docker Daemon Status")
        
        # Check if user can run Docker commands
        run_command("docker ps", "Docker Permissions")
    else:
        print("❌ Docker is not installed. Please install Docker first.")
        return False
    
    return True

def test_nvidia():
    """Test NVIDIA GPU availability"""
    print("\n🔍 NVIDIA GPU Status:")
    
    # Check nvidia-smi
    nvidia_available = run_command("nvidia-smi", "NVIDIA GPU Detection")
    
    if nvidia_available:
        # Check NVIDIA Docker runtime
        run_command("docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi", "NVIDIA Docker Runtime")
    
    return nvidia_available

def test_intel_npu():
    """Test Intel NPU availability"""
    print("\n🔍 Intel NPU Status:")
    
    # Check for NPU devices
    run_command("lspci | grep -i 'neural\\|npu\\|ai\\|accelerator'", "NPU Device Detection")
    
    # Check for Intel graphics/NPU
    run_command("ls /dev/dri/", "Intel Graphics/NPU Devices")
    
    # Check Intel tools
    run_command("ls /opt/intel/ 2>/dev/null || echo 'Intel tools not found'", "Intel Tools")

def test_openvino():
    """Test OpenVINO installation"""
    print("\n🔍 OpenVINO Status:")
    
    try:
        import openvino as ov
        print(f"✓ OpenVINO version: {ov.__version__}")
        
        # Test device detection
        core = ov.Core()
        devices = core.available_devices
        print(f"Available devices: {devices}")
        
        for device in devices:
            try:
                device_name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  {device}: {device_name}")
            except Exception as e:
                print(f"  {device}: Error getting device info - {e}")
        
        return True
    except ImportError:
        print("✗ OpenVINO not installed")
        return False
    except Exception as e:
        print(f"✗ OpenVINO error: {e}")
        return False

def test_face_recognition_models():
    """Test face recognition model files"""
    print("\n🔍 Face Recognition Models:")
    
    # Check if model directories exist
    buffalo_l_path = "/home/sr/ov_fr/buffalo_l"
    if os.path.exists(buffalo_l_path):
        print(f"✓ buffalo_l directory found: {buffalo_l_path}")
        
        # Check SCRFD model
        scrfd_path = os.path.join(buffalo_l_path, "FD/F16/model.xml")
        if os.path.exists(scrfd_path):
            print(f"✓ SCRFD model found: {scrfd_path}")
        else:
            print(f"✗ SCRFD model not found: {scrfd_path}")
        
        # Check AdaFace model
        adaface_path = os.path.join(buffalo_l_path, "Adaface/R50/F32/model.xml")
        if os.path.exists(adaface_path):
            print(f"✓ AdaFace model found: {adaface_path}")
        else:
            print(f"✗ AdaFace model not found: {adaface_path}")
    else:
        print(f"✗ buffalo_l directory not found: {buffalo_l_path}")

def main():
    """Main test function"""
    print("🚀 Local Hardware Test Suite")
    print("=" * 50)
    
    test_system()
    docker_ok = test_docker()
    nvidia_ok = test_nvidia()
    test_intel_npu()
    openvino_ok = test_openvino()
    test_face_recognition_models()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"Docker: {'✓' if docker_ok else '✗'}")
    print(f"NVIDIA GPU: {'✓' if nvidia_ok else '✗'}")
    print(f"OpenVINO: {'✓' if openvino_ok else '✗'}")
    
    if docker_ok:
        print("\n🎯 Next steps:")
        print("1. Run: docker-compose build")
        print("2. Run: docker-compose run --rm face-recognition")
        print("3. Run: docker-compose run --rm face-recognition python3 /app/test_gpu_npu.py")
    else:
        print("\n❌ Please install Docker first")

if __name__ == "__main__":
    main()
