#!/usr/bin/env python3
"""
Test script to verify OpenCV and OpenCL installation and functionality
"""

import sys
import subprocess
import os

def test_opencv_installation():
    """Test OpenCV Python installation"""
    print("=== OpenCV Python Test ===")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV basic operations working")
        
        # Check OpenCV build info
        build_info = cv2.getBuildInformation()
        print("\nOpenCV Build Info:")
        print("-" * 50)
        
        # Look for important build flags
        important_flags = [
            "OpenCL", "CUDA", "Intel", "GPU", "NEON", "AVX", "SSE"
        ]
        
        for line in build_info.split('\n'):
            for flag in important_flags:
                if flag in line:
                    print(f"  {line.strip()}")
                    break
        
        return True
        
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_opencl_installation():
    """Test OpenCL Python installation"""
    print("\n=== OpenCL Python Test ===")
    
    # Test PyOpenCL
    try:
        import pyopencl as cl
        print("✓ PyOpenCL imported successfully")
        
        # Get available platforms
        platforms = cl.get_platforms()
        print(f"✓ Found {len(platforms)} OpenCL platform(s)")
        
        for i, platform in enumerate(platforms):
            print(f"  Platform {i}: {platform.name}")
            print(f"    Vendor: {platform.vendor}")
            print(f"    Version: {platform.version}")
            
            # Get devices for this platform
            devices = platform.get_devices()
            print(f"    Devices: {len(devices)}")
            
            for j, device in enumerate(devices):
                print(f"      Device {j}: {device.name}")
                print(f"        Type: {cl.device_type.to_string(device.type)}")
                print(f"        Compute Units: {device.max_compute_units}")
                print(f"        Global Memory: {device.global_mem_size // (1024*1024)} MB")
        
        return True
        
    except ImportError as e:
        print(f"✗ PyOpenCL import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ OpenCL test failed: {e}")
        return False

def test_opencl4py():
    """Test OpenCL4Py installation"""
    print("\n=== OpenCL4Py Test ===")
    
    try:
        import opencl4py as cl
        print("✓ OpenCL4Py imported successfully")
        
        # Get platforms
        platforms = cl.Platforms()
        print(f"✓ Found {len(platforms)} OpenCL platform(s)")
        
        for i, platform in enumerate(platforms):
            print(f"  Platform {i}: {platform.name}")
            devices = platform.devices
            print(f"    Devices: {len(devices)}")
            
            for j, device in enumerate(devices):
                print(f"      Device {j}: {device.name}")
                print(f"        Type: {device.type}")
        
        return True
        
    except ImportError as e:
        print(f"✗ OpenCL4Py import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ OpenCL4Py test failed: {e}")
        return False

def test_system_opencl():
    """Test system OpenCL installation"""
    print("\n=== System OpenCL Test ===")
    
    try:
        # Test clinfo command
        result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ clinfo command working")
            print("OpenCL Info:")
            print("-" * 30)
            
            # Parse clinfo output for important info
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['platform', 'device', 'intel', 'nvidia', 'amd']):
                    print(f"  {line.strip()}")
        else:
            print(f"✗ clinfo failed: {result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ clinfo command timed out")
        return False
    except FileNotFoundError:
        print("✗ clinfo command not found")
        return False
    except Exception as e:
        print(f"✗ System OpenCL test failed: {e}")
        return False

def test_opencv_opencl_integration():
    """Test OpenCV with OpenCL backend"""
    print("\n=== OpenCV-OpenCL Integration Test ===")
    
    try:
        import cv2
        import numpy as np
        
        # Check if OpenCL is available in OpenCV
        if cv2.ocl.haveOpenCL():
            print("✓ OpenCV has OpenCL support")
            
            # Enable OpenCL
            cv2.ocl.setUseOpenCL(True)
            print("✓ OpenCL enabled in OpenCV")
            
            # Test OpenCL device info
            devices = cv2.ocl.Device.getDefault()
            if devices:
                print(f"✓ Default OpenCL device: {devices.name()}")
                print(f"  Vendor: {devices.vendorName()}")
                print(f"  Version: {devices.version()}")
                print(f"  Compute Units: {devices.maxComputeUnits()}")
                print(f"  Global Memory: {devices.globalMemSize() // (1024*1024)} MB")
            
            # Test basic OpenCL operation
            img = np.random.randn(100, 100, 3).astype(np.float32)
            
            # This should use OpenCL if available
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            print("✓ OpenCV OpenCL operations working")
            return True
        else:
            print("✗ OpenCV does not have OpenCL support")
            return False
            
    except Exception as e:
        print(f"✗ OpenCV-OpenCL integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("OpenCV and OpenCL Installation Test")
    print("=" * 50)
    
    tests = [
        test_opencv_installation,
        test_opencl_installation,
        test_opencl4py,
        test_system_opencl,
        test_opencv_opencl_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"OpenCV: {'✓' if results[0] else '✗'}")
    print(f"PyOpenCL: {'✓' if results[1] else '✗'}")
    print(f"OpenCL4Py: {'✓' if results[2] else '✗'}")
    print(f"System OpenCL: {'✓' if results[3] else '✗'}")
    print(f"OpenCV-OpenCL Integration: {'✓' if results[4] else '✗'}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✓ All tests passed' if all_passed else '✗ Some tests failed'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
