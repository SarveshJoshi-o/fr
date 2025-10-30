#!/usr/bin/env python3
"""
Test script to verify GPU and NPU access in Docker container
"""

import os
import sys
import subprocess
import time

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

def test_system_info():
    """Test basic system information"""
    print("🔍 Testing System Information...")
    
    # OS info
    run_command("cat /etc/os-release | grep PRETTY_NAME", "OS Version")
    
    # Python version
    run_command("python3 --version", "Python Version")
    
    # Available memory
    run_command("free -h", "Memory Information")
    
    # CPU info
    run_command("lscpu | grep 'Model name'", "CPU Information")

def test_openvino():
    """Test OpenVINO installation and device detection"""
    print("\n🔍 Testing OpenVINO...")
    
    # Test OpenVINO import
    run_command("python3 -c 'import openvino; print(f\"OpenVINO version: {openvino.__version__}\")'", "OpenVINO Import")
    
    # Test device detection
    run_command("python3 -c \"import openvino as ov; core = ov.Core(); print('Available devices:', core.available_devices)\"", "OpenVINO Device Detection")
    
    # Test specific device properties
    run_command("python3 -c \"import openvino as ov; core = ov.Core(); devices = core.available_devices; [print(f'{d}: {core.get_property(d, \\\"FULL_DEVICE_NAME\\\")}') for d in devices]\"", "Device Properties")

def test_nvidia_gpu():
    """Test NVIDIA GPU access"""
    print("\n🔍 Testing NVIDIA GPU...")
    
    # Check if nvidia-smi is available
    run_command("which nvidia-smi", "NVIDIA SMI Availability")
    
    # Get GPU information
    run_command("nvidia-smi", "NVIDIA GPU Information")
    
    # Test CUDA availability
    run_command("python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")' 2>/dev/null || echo 'PyTorch not available'", "CUDA Availability")

def test_intel_npu():
    """Test Intel NPU access"""
    print("\n🔍 Testing Intel NPU...")
    
    # Check for NPU devices
    run_command("lspci | grep -i 'neural\\|npu\\|ai\\|accelerator'", "NPU Device Detection")
    
    # Check for Intel NPU specific devices
    run_command("ls /dev/dri/", "Intel Graphics/NPU Devices")
    
    # Check for Intel NPU runtime
    run_command("ls /opt/intel/ 2>/dev/null || echo 'Intel tools not found'", "Intel Tools")

def test_openvino_devices():
    """Test OpenVINO device capabilities"""
    print("\n🔍 Testing OpenVINO Device Capabilities...")
    
    # Create a simple test script
    test_script = '''
import openvino as ov
import numpy as np

try:
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    
    for device in devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"Device: {device}")
            print(f"  Name: {device_name}")
            
            # Test device capabilities
            try:
                capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
                print(f"  Capabilities: {capabilities}")
            except:
                print("  Capabilities: Not available")
                
        except Exception as e:
            print(f"  Error getting info for {device}: {e}")
            
    # Test creating a simple model
    print("\\nTesting model creation...")
    try:
        # Create a simple model for testing
        from openvino import Model, Type, Shape
        from openvino.runtime import opset10 as ops
        
        # Create a simple model
        param = ops.parameter(Shape([1, 3, 224, 224]), Type.f32, "input")
        result = ops.softmax(param, axis=1)
        model = Model([result], [param], "test_model")
        
        # Compile model on each device
        for device in devices:
            try:
                compiled_model = core.compile_model(model, device)
                print(f"✓ Model compiled successfully on {device}")
            except Exception as e:
                print(f"✗ Failed to compile model on {device}: {e}")
                
    except Exception as e:
        print(f"Model creation test failed: {e}")
        
except Exception as e:
    print(f"OpenVINO test failed: {e}")
'''
    
    with open("/tmp/test_openvino.py", "w") as f:
        f.write(test_script)
    
    run_command("python3 /tmp/test_openvino.py", "OpenVINO Device Testing")

def test_face_recognition_models():
    """Test if face recognition models can be loaded"""
    print("\n🔍 Testing Face Recognition Models...")
    
    # Check if model files exist
    run_command("ls -la /app/buffalo_l/", "Model Directory Contents")
    
    # Test model loading
    test_model_script = '''
import sys
import os
sys.path.append('/app')

try:
    from fd.scrfd_openvino_sd_blur_detect import SCRFD
    from fr.adaface_openvino import AdaFaceOpenVINO
    
    print("Testing model loading...")
    
    # Test SCRFD model
    detector_path = "/app/buffalo_l/FD/F16/model.xml"
    if os.path.exists(detector_path):
        print(f"✓ SCRFD model found: {detector_path}")
        try:
            detector = SCRFD(detector_path)
            print("✓ SCRFD model loaded successfully")
        except Exception as e:
            print(f"✗ SCRFD model loading failed: {e}")
    else:
        print(f"✗ SCRFD model not found: {detector_path}")
    
    # Test AdaFace model
    adaface_path = "/app/buffalo_l/Adaface/R50/F32/model.xml"
    if os.path.exists(adaface_path):
        print(f"✓ AdaFace model found: {adaface_path}")
        try:
            rec = AdaFaceOpenVINO(adaface_path)
            print("✓ AdaFace model loaded successfully")
        except Exception as e:
            print(f"✗ AdaFace model loading failed: {e}")
    else:
        print(f"✗ AdaFace model not found: {adaface_path}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
'''
    
    with open("/tmp/test_models.py", "w") as f:
        f.write(test_model_script)
    
    run_command("python3 /tmp/test_models.py", "Face Recognition Model Testing")

def main():
    """Main test function"""
    print("🚀 Starting GPU/NPU Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_system_info()
    test_openvino()
    test_nvidia_gpu()
    test_intel_npu()
    test_openvino_devices()
    test_face_recognition_models()
    
    print("\n" + "=" * 50)
    print("🏁 Test Suite Complete")
    print("\nTo run your face recognition application:")
    print("python3 /app/main.py")

if __name__ == "__main__":
    main()
