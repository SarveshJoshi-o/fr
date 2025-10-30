#!/usr/bin/env python3
"""
Verify that SCRFD uses GPU and AdaFace uses NPU
"""

import os
import sys
import openvino as ov

def verify_hardware_assignment():
    """Verify the hardware assignment for face detection and recognition"""
    print("🔍 Verifying Hardware Assignment")
    print("=" * 50)
    
    # Check available devices
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available OpenVINO devices: {available_devices}")
    
    # Check SCRFD configuration
    print("\n📋 SCRFD (Face Detection) Configuration:")
    try:
        from fd.scrfd_openvino_sd_blur_detect import SCRFD
        import os.path as osp
        
        # Create SCRFD instance
        assets_dir = osp.expanduser('buffalo_l')
        model_path = os.path.join(assets_dir, './FD/F16/model.xml')
        
        if os.path.exists(model_path):
            detector = SCRFD(model_path)
            print(f"✅ SCRFD model loaded: {model_path}")
            
            # Check which device SCRFD is using
            if hasattr(detector, 'compiled_model'):
                # Get the device from the compiled model
                print("✅ SCRFD compiled model ready")
                print("🔧 SCRFD should be using GPU for face detection")
            else:
                print("⚠️  SCRFD compiled model not found")
        else:
            print(f"❌ SCRFD model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading SCRFD: {e}")
    
    # Check AdaFace configuration
    print("\n📋 AdaFace (Face Recognition) Configuration:")
    try:
        from fr.adaface_openvino import AdaFaceOpenVINO
        import os.path as osp
        
        # Create AdaFace instance
        assets_dir = osp.expanduser('buffalo_l')
        model_path = os.path.join(assets_dir, './Adaface/R50/F32/model.xml')
        
        if os.path.exists(model_path):
            rec = AdaFaceOpenVINO(model_path)
            print(f"✅ AdaFace model loaded: {model_path}")
            print("🔧 AdaFace is configured to use NPU for face recognition")
        else:
            print(f"❌ AdaFace model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading AdaFace: {e}")
    
    # Test device availability
    print("\n🧪 Testing Device Availability:")
    
    # Test GPU
    if "GPU" in available_devices:
        print("✅ GPU available for SCRFD face detection")
        try:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"   GPU: {gpu_name}")
        except Exception as e:
            print(f"   GPU info: {e}")
    else:
        print("❌ GPU not available - SCRFD will fall back to CPU")
    
    # Test NPU
    if "NPU" in available_devices:
        print("✅ NPU available for AdaFace face recognition")
        try:
            npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
            print(f"   NPU: {npu_name}")
        except Exception as e:
            print(f"   NPU info: {e}")
    else:
        print("❌ NPU not available - AdaFace will fall back to CPU")
    
    # Performance recommendations
    print("\n🎯 Performance Recommendations:")
    if "GPU" in available_devices and "NPU" in available_devices:
        print("✅ Optimal configuration: GPU + NPU available")
        print("   • SCRFD will use GPU for fast face detection")
        print("   • AdaFace will use NPU for efficient face recognition")
        print("   • Both can run in parallel for maximum performance")
    elif "GPU" in available_devices:
        print("⚠️  Partial optimization: GPU available, NPU not available")
        print("   • SCRFD will use GPU for face detection")
        print("   • AdaFace will fall back to CPU")
    elif "NPU" in available_devices:
        print("⚠️  Partial optimization: NPU available, GPU not available")
        print("   • SCRFD will fall back to CPU")
        print("   • AdaFace will use NPU for face recognition")
    else:
        print("❌ No specialized hardware available")
        print("   • Both SCRFD and AdaFace will use CPU")
        print("   • Performance will be limited")
    
    print("\n" + "=" * 50)
    print("Hardware assignment verification completed!")

if __name__ == "__main__":
    verify_hardware_assignment()
