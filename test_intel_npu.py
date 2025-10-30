#!/usr/bin/env python3
"""
Test script to verify Intel NPU support in OpenVINO
"""

import os
import subprocess
import sys

def check_intel_npu_support():
    """Check if Intel NPU is available and working"""
    print("=== Intel NPU Support Test ===")
    
    # Check for NPU devices
    if os.path.exists('/dev/accel0'):
        print("✓ /dev/accel0 device found")
        try:
            stat_info = os.stat('/dev/accel0')
            print(f"  accel0: mode={oct(stat_info.st_mode)}, uid={stat_info.st_uid}, gid={stat_info.st_gid}")
        except Exception as e:
            print(f"✗ Error checking accel0 device: {e}")
    else:
        print("✗ /dev/accel0 device not found")
        print("  Intel NPU may not be available or drivers not installed")
    
    # Check for Intel NPU packages
    try:
        result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            npu_packages = []
            for line in result.stdout.split('\n'):
                if 'intel' in line.lower() and ('npu' in line.lower() or 'level-zero' in line.lower()):
                    npu_packages.append(line.strip())
            
            if npu_packages:
                print("✓ Intel NPU packages found:")
                for package in npu_packages:
                    print(f"  {package}")
            else:
                print("✗ No Intel NPU packages found")
        else:
            print("✗ Error checking installed packages")
    except Exception as e:
        print(f"✗ Package check failed: {e}")
    
    # Check Level Zero runtime
    try:
        result = subprocess.run(['ls', '/usr/lib/x86_64-linux-gnu/'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            level_zero_libs = [lib for lib in result.stdout.split('\n') if 'level' in lib.lower() and 'zero' in lib.lower()]
            if level_zero_libs:
                print("✓ Level Zero libraries found:")
                for lib in level_zero_libs:
                    print(f"  {lib}")
            else:
                print("✗ No Level Zero libraries found")
        else:
            print("✗ Error checking Level Zero libraries")
    except Exception as e:
        print(f"✗ Level Zero check failed: {e}")

def test_openvino_npu_support():
    """Test OpenVINO NPU support"""
    print("\n=== OpenVINO NPU Test ===")
    
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        print(f"Available devices: {available_devices}")
        
        # Look for NPU device
        npu_devices = [device for device in available_devices if 'NPU' in device.upper()]
        if npu_devices:
            print(f"✓ NPU devices found: {npu_devices}")
            for device in npu_devices:
                try:
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    print(f"  {device}: {device_name}")
                except Exception as e:
                    print(f"  {device}: Error getting device info - {e}")
        else:
            print("✗ No NPU devices found in OpenVINO")
            print("  This might be because:")
            print("  1. Intel NPU drivers not installed")
            print("  2. NPU device not available on this system")
            print("  3. OpenVINO NPU plugin not available")
        
        return len(npu_devices) > 0
        
    except ImportError:
        print("✗ OpenVINO not available")
        return False
    except Exception as e:
        print(f"✗ OpenVINO NPU test failed: {e}")
        return False

def test_npu_inference():
    """Test NPU inference if available"""
    print("\n=== NPU Inference Test ===")
    
    try:
        import openvino as ov
        import numpy as np
        
        core = ov.Core()
        available_devices = core.available_devices
        
        # Look for NPU device
        npu_device = None
        for device in available_devices:
            if 'NPU' in device.upper():
                npu_device = device
                break
        
        if not npu_device:
            print("✗ No NPU device available for testing")
            return False
        
        print(f"✓ Testing with NPU device: {npu_device}")
        
        # Create a simple test model (identity function)
        from openvino.runtime import Model, op
        from openvino.runtime import opset10 as ops
        
        # Create a simple model: y = x
        param = ops.parameter([1, 3, 224, 224], ov.Type.f32, name="input")
        result = ops.result(param, name="output")
        model = Model([result], [param], "test_model")
        
        # Compile model for NPU
        compiled_model = core.compile_model(model, npu_device)
        
        # Test inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = compiled_model([input_data])
        
        print("✓ NPU inference test successful!")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {result[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ NPU inference test failed: {e}")
        return False

def main():
    """Run all NPU tests"""
    print("Intel NPU Support Test for OpenVINO")
    print("=" * 50)
    
    # Check NPU support
    check_intel_npu_support()
    
    # Test OpenVINO NPU support
    npu_available = test_openvino_npu_support()
    
    # Test NPU inference if available
    if npu_available:
        test_npu_inference()
    
    print("\n" + "=" * 50)
    print("NPU Test completed!")
    
    return npu_available

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
