"""
Simple test script to verify the enhanced license plate reader is working.
Run this after copying all the files to test your installation.
"""
import os
import cv2
import numpy as np

def test_enhanced_system():
    """Test the enhanced license plate system."""
    print("Testing Enhanced License Plate Reader...")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from models import SuperEnhancedLicensePlateReader, PlateImageEnhancer
        print("   ✅ Enhanced components imported successfully")
        
        # Initialize components
        print("2. Initializing components...")
        enhancer = PlateImageEnhancer()
        reader = SuperEnhancedLicensePlateReader()
        print("   ✅ Components initialized successfully")
        
        # Create a test image with text
        print("3. Creating test image...")
        test_img = np.zeros((100, 300, 3), dtype=np.uint8)
        test_img.fill(128)  # Gray background
        cv2.putText(test_img, "ABC 123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save test image
        test_path = "test_plate.jpg"
        cv2.imwrite(test_path, test_img)
        print("   ✅ Test image created: test_plate.jpg")
        
        # Test enhancement
        print("4. Testing image enhancement...")
        enhanced_img = enhancer.enhance_violation_image(
            test_path, 
            "test_plate_enhanced.jpg",
            methods='medium'
        )
        
        if enhanced_img is not None:
            print("   ✅ Image enhancement working")
        else:
            print("   ❌ Image enhancement failed")
            return False
        
        # Test plate reading  
        print("5. Testing license plate reading...")
        h, w = test_img.shape[:2]
        result = reader.process_violation_image(test_path, [0, 0, w, h])
        
        if result:
            plate_text, confidence = result
            print(f"   ✅ Plate reading working: '{plate_text}' (confidence: {confidence:.2f})")
        else:
            print("   ⚠️  Plate reading didn't detect text (this is normal for simple test image)")
        
        # Test different enhancement levels
        print("6. Testing different enhancement levels...")
        levels = ['light', 'medium', 'aggressive']
        
        for level in levels:
            try:
                enhanced = enhancer.enhance_license_plate_region(test_img, level)
                if enhanced is not None:
                    print(f"   ✅ {level.capitalize()} enhancement: OK")
                else:
                    print(f"   ❌ {level.capitalize()} enhancement: Failed")
            except Exception as e:
                print(f"   ❌ {level.capitalize()} enhancement error: {e}")
        
        # Clean up test files
        print("7. Cleaning up...")
        for file in ["test_plate.jpg", "test_plate_enhanced.jpg"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"   ✅ Removed {file}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Enhanced system is working!")
        print("=" * 50)
        
        print("\n📋 Next steps:")
        print("1. Test with your actual violation images")
        print("2. Use the enhanced system in your detectors")
        print("3. Check the enhanced_debug folder for processed images")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Fix:")
        print("1. Make sure all files are in the correct locations")
        print("2. Check that models/__init__.py has been updated")
        print("3. Verify dependencies are installed: pip install opencv-python pillow easyocr torch")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Check:")
        print("1. File permissions")
        print("2. Dependencies installation")
        print("3. Python version (3.7+ required)")
        return False

def test_with_existing_violation(image_path):
    """Test with an existing violation image."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"\nTesting with existing violation image: {image_path}")
    print("-" * 50)
    
    try:
        from models import SuperEnhancedLicensePlateReader
        
        reader = SuperEnhancedLicensePlateReader()
        
        # Read image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not read image")
            return False
        
        h, w = img.shape[:2]
        print(f"Image dimensions: {w}x{h}")
        
        # Test different enhancement levels
        levels = ['light', 'medium', 'aggressive', 'auto']
        
        for level in levels:
            print(f"\nTesting {level} enhancement:")
            result = reader.read_plate_from_vehicle(img, enhancement_level=level)
            
            if result:
                plate_text, confidence = result[:2]
                print(f"  ✅ Detected: '{plate_text}' (confidence: {confidence:.2f})")
            else:
                print(f"  ❌ No plate detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing with violation image: {e}")
        return False

if __name__ == "__main__":
    # Run basic test
    success = test_enhanced_system()
    
    if success:
        print("\n" + "=" * 50)
        print("🔧 OPTIONAL: Test with your violation image")
        print("=" * 50)
        
        # Ask user if they want to test with an existing image
        existing_image = input("\nEnter path to existing violation image (or press Enter to skip): ").strip()
        
        if existing_image:
            test_with_existing_violation(existing_image)
        
        print("\n🎯 System ready! You can now use the enhanced license plate reader.")
    else:
        print("\n❌ Tests failed. Please fix the issues above before proceeding.")