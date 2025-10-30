#!/usr/bin/env python3
"""
Test script to verify Intel GPU support in OpenVINO
"""

import os
import subprocess
import sys

def check_intel_gpu_support():
    """Check if Intel GPU is available and working"""
    print("=== Intel GPU Support Test ===")
    
    # Check if we're in a container with proper device access
    if os.path.exists('/dev/dri'):
        print("✓ /dev/dri directory found")
        try:
            dri_devices = os.listdir('/dev/dri')
            print(f"✓ DRI devices available: {dri_devices}")
            
            # Check device permissions
            for device in dri_devices:
                device_path = f"/dev/dri/{device}"
                if os.path.exists(device_path):
                    stat_info = os.stat(device_path)
                    print(f"  {device}: mode={oct(stat_info.st_mode)}, uid={stat_info.st_uid}, gid={stat_info.st_gid}")
        except Exception as e:
            print(f"✗ Error listing DRI devices: {e}")
    else:
        print("✗ /dev/dri directory not found - Intel GPU may not be accessible")
        print("  Make sure to run with: --device /dev/dri:/dev/dri")
    
    # Check VA-API support
    try:
        result = subprocess.run(['vainfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ VA-API is working")
            print("VA-API Info:")
            print(result.stdout)
        else:
            print("✗ VA-API not working")
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ VA-API test timed out")
    except FileNotFoundError:
        print("✗ vainfo command not found")
    except Exception as e:
        print(f"✗ VA-API test failed: {e}")
    
    # Check OpenVINO GPU plugin
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        print(f"\n=== OpenVINO Available Devices ===")
        print(f"Available devices: {available_devices}")
        
        # Check specifically for GPU
        gpu_devices = [device for device in available_devices if 'GPU' in device.upper()]
        if gpu_devices:
            print(f"✓ GPU devices found: {gpu_devices}")
            for device in gpu_devices:
                try:
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    print(f"  {device}: {device_name}")
                except Exception as e:
                    print(f"  {device}: Error getting device info - {e}")
        else:
            print("✗ No GPU devices found in OpenVINO")
            print("  This might be because:")
            print("  1. Intel GPU drivers not properly installed")
            print("  2. /dev/dri devices not mounted in container")
            print("  3. OpenVINO GPU plugin not available")
            
        # Check for Intel GPU specifically
        intel_gpu_devices = [device for device in available_devices if 'GPU' in device.upper()]
        if intel_gpu_devices:
            print(f"✓ Intel GPU devices: {intel_gpu_devices}")
        else:
            print("✗ No Intel GPU devices detected")
            print("  Try running with: --device /dev/dri:/dev/dri --privileged")
            
    except ImportError:
        print("✗ OpenVINO not available")
    except Exception as e:
        print(f"✗ OpenVINO test failed: {e}")
    
    # Check environment variables
    print(f"\n=== Environment Variables ===")
    print(f"LIBVA_DRIVER_NAME: {os.environ.get('LIBVA_DRIVER_NAME', 'Not set')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    
    # Check for Intel GPU in lspci
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            intel_gpu_lines = [line for line in result.stdout.split('\n') if 'Intel' in line and ('Graphics' in line or 'VGA' in line)]
            if intel_gpu_lines:
                print(f"\n=== Intel GPU Hardware ===")
                for line in intel_gpu_lines:
                    print(f"✓ {line}")
            else:
                print("\n✗ No Intel GPU found in lspci")
        else:
            print("✗ lspci command failed")
    except Exception as e:
        print(f"✗ lspci test failed: {e}")

def test_openvino_gpu_inference():
    """Test actual OpenVINO GPU inference"""
    print("\n=== OpenVINO GPU Inference Test ===")
    
    try:
        import openvino as ov
        import numpy as np
        
        core = ov.Core()
        available_devices = core.available_devices
        
        # Look for GPU device
        gpu_device = None
        for device in available_devices:
            if 'GPU' in device.upper():
                gpu_device = device
                break
        
        if not gpu_device:
            print("✗ No GPU device available for testing")
            return
        
        print(f"✓ Testing with device: {gpu_device}")
        
        # Create a simple test model (identity function)
        from openvino.runtime import Model, op
        from openvino.runtime import opset10 as ops
        
        # Create a simple model: y = x
        param = ops.parameter([1, 3, 224, 224], ov.Type.f32, name="input")
        result = ops.result(param, name="output")
        model = Model([result], [param], "test_model")
        
        # Compile model for GPU
        compiled_model = core.compile_model(model, gpu_device)
        
        # Test inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = compiled_model([input_data])
        
        print("✓ GPU inference test successful!")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {result[0].shape}")
        
    except Exception as e:
        print(f"✗ GPU inference test failed: {e}")

if __name__ == "__main__":
    print("Intel GPU Support Test for OpenVINO")
    print("=" * 50)
    
    check_intel_gpu_support()
    test_openvino_gpu_inference()
    
    print("\n" + "=" * 50)
    print("Test completed!")
