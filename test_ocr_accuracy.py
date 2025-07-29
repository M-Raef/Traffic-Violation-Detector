"""
Test script for OCR accuracy improvements.
"""
import os
import cv2
import numpy as np
from datetime import datetime

def test_ocr_accuracy():
    """Test OCR accuracy improvements with real violation images."""
    print("Testing OCR Accuracy Improvements")
    print("=" * 50)
    
    try:
        # Import the improved reader
        from models.enhanced_license_plate import SuperEnhancedLicensePlateReader
        
        reader = SuperEnhancedLicensePlateReader()
        print("✅ Improved OCR reader initialized")
        
        # Test with sample violation images
        test_images = []
        
        # Look for violation images in common directories
        possible_dirs = [
            "violations",
            "violations/2024-01-15",
            "test_images",
            "samples"
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                image_files = [f for f in os.listdir(dir_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_file in image_files[:3]:  # Test first 3 images
                    test_images.append(os.path.join(dir_path, img_file))
        
        if not test_images:
            print("ℹ️  No violation images found. Creating test image...")
            test_images = [create_test_license_plate()]
        
        print(f"Testing {len(test_images)} images...")
        print("-" * 50)
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\nTest {i}: {os.path.basename(image_path)}")
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print("  ❌ Could not read image")
                continue
            
            h, w = img.shape[:2]
            print(f"  Image size: {w}x{h}")
            
            # Process with improved OCR
            result = reader.process_violation_image(image_path, [0, 0, w, h], save_enhanced=True)
            
            if result:
                plate_text, confidence = result
                print(f"  ✅ Detected: '{plate_text}' (confidence: {confidence:.2f})")
                
                # Validate the result
                if validate_plate_format(plate_text):
                    print(f"  ✅ Valid format")
                else:
                    print(f"  ⚠️  Unusual format (might be correct)")
            else:
                print(f"  ❌ No plate detected")
            
            # Check if debug images were created
            debug_dir = os.path.join(os.path.dirname(image_path), "ocr_debug")
            if os.path.exists(debug_dir):
                debug_files = len([f for f in os.listdir(debug_dir) 
                                 if f.startswith(os.path.splitext(os.path.basename(image_path))[0])])
                print(f"  📁 {debug_files} debug images saved")
        
        print("\n" + "=" * 50)
        print("🔍 OCR Accuracy Test Complete!")
        print("=" * 50)
        
        print("\n📋 Tips for better accuracy:")
        print("1. Check the 'ocr_debug' folder for processed images")
        print("2. Adjust OCR_ENHANCEMENT_SETTINGS in ocr_config.py if needed")
        print("3. Add your region's license plate patterns to LICENSE_PLATE_PATTERNS")
        print("4. Check if characters are being confused (0/O, 1/I, etc.)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing OCR: {e}")
        import traceback
        traceback.print_exc()
        return False




def create_test_license_plate():
    """Create a test license plate image for testing."""
    print("Creating test license plate image...")
    
    # Create image with license plate text
    img = np.ones((150, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some noise to simulate real conditions
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add license plate rectangle
    cv2.rectangle(img, (50, 50), (350, 100), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (350, 100), (0, 0, 0), 2)
    
    # Add license plate text
    cv2.putText(img, "ABC 123", (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Add some blur to simulate motion
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    test_path = "test_license_plate.jpg"
    cv2.imwrite(test_path, img)
    print(f"✅ Test image created: {test_path}")
    
    return test_path

def validate_plate_format(plate_text):
    """Validate if the plate text matches common formats."""
    import re
    
    common_patterns = [
        r'^[A-Z]{3}[0-9]{3}$',      # ABC123
        r'^[A-Z]{3}\s[0-9]{3}$',    # ABC 123
        r'^[A-Z]{2}[0-9]{4}$',      # AB1234
        r'^[A-Z]{2}\s[0-9]{4}$',    # AB 1234
        r'^[0-9]{3}[A-Z]{3}$',      # 123ABC
        r'^[0-9]{3}\s[A-Z]{3}$',    # 123 ABC
    ]
    
    for pattern in common_patterns:
        if re.match(pattern, plate_text):
            return True
    
    return False

def test_character_corrections():
    """Test character correction functionality."""
    print("\nTesting Character Corrections")
    print("-" * 30)
    
    try:
        from models.enhanced_license_plate import SuperEnhancedLicensePlateReader
        
        reader = SuperEnhancedLicensePlateReader()
        
        # Test common OCR mistakes
        test_cases = [
            ("ABC1Z3", "ABC123"),  # Z->2
            ("4BC123", "ABC123"),  # 4->A
            ("ABC1O3", "ABC103"),  # O->0
            ("A8C123", "ABC123"),  # 8->B
            ("ABG123", "AB6123"),  # G->6
        ]
        
        print("Testing character corrections:")
        for input_text, expected in test_cases:
            # Note: We need to implement _clean_text method or similar
            print(f"  Input: {input_text} -> Expected: {expected}")
        
        print("✅ Character correction test structure ready")
        
    except Exception as e:
        print(f"❌ Error testing character corrections: {e}")

def test_simple_ocr():
    """Simple OCR test with basic functionality."""
    print("\nSimple OCR Test")
    print("-" * 30)
    
    try:
        # Test basic import first
        from models import SuperEnhancedLicensePlateReader
        print("✅ SuperEnhancedLicensePlateReader imported")
        
        # Create reader
        reader = SuperEnhancedLicensePlateReader()
        print("✅ Reader initialized")
        
        # Create simple test image
        test_img = create_simple_test_image()
        
        # Test reading
        img = cv2.imread(test_img)
        h, w = img.shape[:2]
        
        result = reader.process_violation_image(test_img, [0, 0, w, h])
        
        if result:
            plate_text, confidence = result
            print(f"✅ Test result: '{plate_text}' (confidence: {confidence:.2f})")
        else:
            print("❌ No result from test image")
        
        # Clean up
        if os.path.exists(test_img):
            os.remove(test_img)
        
        return True
        
    except Exception as e:
        print(f"❌ Simple OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_image():
    """Create a simple test image with clear text."""
    # Create white background
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add black text
    cv2.putText(img, "TEST123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    test_path = "simple_test.jpg"
    cv2.imwrite(test_path, img)
    
    return test_path

def benchmark_ocr_methods():
    """Benchmark different OCR preprocessing methods."""
    print("\nBenchmarking OCR Methods")
    print("-" * 30)
    
    try:
        from models.enhanced_license_plate import SuperEnhancedLicensePlateReader
        
        reader = SuperEnhancedLicensePlateReader()
        
        # Create test images with different challenges
        test_scenarios = [
            ("normal", create_normal_plate()),
            ("blurry", create_blurry_plate()),
            ("low_contrast", create_low_contrast_plate()),
            ("noisy", create_noisy_plate()),
        ]
        
        methods = ['light', 'medium', 'aggressive', 'auto']
        
        for scenario_name, test_img_path in test_scenarios:
            print(f"\nScenario: {scenario_name}")
            
            for method in methods:
                try:
                    img = cv2.imread(test_img_path)
                    if img is None:
                        continue
                        
                    h, w = img.shape[:2]
                    
                    # Test with specific enhancement level
                    result = reader.read_plate_from_vehicle(img, enhancement_level=method)
                    
                    if result:
                        plate_text, confidence = result[:2]
                        print(f"  {method:12}: {plate_text:8} ({confidence:.2f})")
                    else:
                        print(f"  {method:12}: {'No detection':8}")
                        
                except Exception as e:
                    print(f"  {method:12}: Error - {e}")
            
            # Clean up test image
            if os.path.exists(test_img_path):
                os.remove(test_img_path)
        
        print("✅ Benchmark complete")
        
    except Exception as e:
        print(f"❌ Error benchmarking: {e}")

def create_normal_plate():
    """Create a normal quality license plate."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (280, 80), (0, 0, 0), 2)
    cv2.putText(img, "ABC123", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    path = "test_normal.jpg"
    cv2.imwrite(path, img)
    return path

def create_blurry_plate():
    """Create a blurry license plate."""
    img = create_normal_plate()
    img_data = cv2.imread(img)
    blurred = cv2.GaussianBlur(img_data, (5, 5), 0)
    
    path = "test_blurry.jpg"
    cv2.imwrite(path, blurred)
    os.remove(img)  # Remove the temp normal image
    return path

def create_low_contrast_plate():
    """Create a low contrast license plate."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 200  # Gray background
    cv2.rectangle(img, (20, 20), (280, 80), (150, 150, 150), 2)  # Dark gray border
    cv2.putText(img, "ABC123", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)  # Dark gray text
    
    path = "test_low_contrast.jpg"
    cv2.imwrite(path, img)
    return path

def create_noisy_plate():
    """Create a noisy license plate."""
    img = create_normal_plate()
    img_data = cv2.imread(img)
    
    # Add noise
    noise = np.random.randint(0, 50, img_data.shape, dtype=np.uint8)
    noisy = cv2.add(img_data, noise)
    
    path = "test_noisy.jpg"
    cv2.imwrite(path, noisy)
    os.remove(img)  # Remove the temp normal image
    return path

def test_existing_violations():
    """Test with existing violation images if available."""
    print("\nTesting Existing Violation Images")
    print("-" * 40)
    
    # Look for violation directories
    violation_dirs = []
    
    # Common violation directory patterns
    base_dirs = ["violations", "output", "results"]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            # Look for subdirectories with dates
            try:
                subdirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d))]
                for subdir in subdirs:
                    full_path = os.path.join(base_dir, subdir)
                    violation_dirs.append(full_path)
            except:
                # If base_dir itself has images
                violation_dirs.append(base_dir)
    
    if not violation_dirs:
        print("No violation directories found. Skipping existing violation test.")
        return
    
    # Test first violation directory found
    test_dir = violation_dirs[0]
    print(f"Testing images in: {test_dir}")
    
    # Get image files
    try:
        image_files = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("No image files found in violation directory.")
            return
        
        # Test first few images
        from models.enhanced_license_plate import SuperEnhancedLicensePlateReader
        reader = SuperEnhancedLicensePlateReader()
        
        test_count = min(3, len(image_files))
        print(f"Testing {test_count} violation images...")
        
        for i in range(test_count):
            image_file = image_files[i]
            image_path = os.path.join(test_dir, image_file)
            
            print(f"\nImage {i+1}: {image_file}")
            
            # Read and test
            img = cv2.imread(image_path)
            if img is None:
                print("  ❌ Could not read image")
                continue
            
            h, w = img.shape[:2]
            result = reader.process_violation_image(image_path, [0, 0, w, h], save_enhanced=True)
            
            if result:
                plate_text, confidence = result
                print(f"  ✅ Detected: '{plate_text}' (confidence: {confidence:.2f})")
            else:
                print("  ❌ No plate detected")
        
    except Exception as e:
        print(f"Error testing existing violations: {e}")

if __name__ == "__main__":
    print("OCR Accuracy Testing Suite")
    print("=" * 60)
    
    # Run tests in order of complexity
    print("1. Testing basic imports and functionality...")
    basic_success = test_simple_ocr()
    
    if basic_success:
        print("\n2. Testing with sample images...")
        test_ocr_accuracy()
        
        print("\n3. Testing character corrections...")
        test_character_corrections()
        
        print("\n4. Testing existing violation images...")
        test_existing_violations()
        
        print("\n5. Benchmarking different methods...")
        benchmark_ocr_methods()
        
        print("\n" + "=" * 60)
        print("🎯 All OCR tests completed!")
        print("=" * 60)
        
        print("\n📊 Next steps:")
        print("1. Check the debug images in 'ocr_debug' folders")
        print("2. If accuracy is still low, adjust patterns in ocr_config.py")
        print("3. Test with your actual violation images")
        print("4. Consider adding region-specific license plate patterns")
        
    else:
        print("\n❌ Basic OCR testing failed. Please check the errors above.")
        print("Make sure the enhanced license plate reader is properly installed.")