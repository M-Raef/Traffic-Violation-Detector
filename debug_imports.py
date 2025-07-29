"""
Debug script to test imports step by step.
"""

def test_individual_imports():
    """Test each import individually to find the issue."""
    print("Testing individual imports...")
    print("=" * 40)
    
    # Test 1: Basic models directory
    try:
        import models
        print("✅ models package imported")
    except Exception as e:
        print(f"❌ Cannot import models package: {e}")
        return False
    
    # Test 2: Individual model files
    imports_to_test = [
        ('models.detector', 'YOLODetector'),
        ('models.tracker', 'StableTracker'),
        ('models.traffic_light', 'TrafficLightAnalyzer'),
        ('models.license_plate', 'EnhancedLicensePlateReader'),
    ]
    
    working_imports = []
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            working_imports.append((module_name, class_name))
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
    
    # Test 3: Check if new files exist
    import os
    files_to_check = [
        'models/plate_enhancer.py',
        'models/enhanced_license_plate.py'
    ]
    
    print("\nChecking new files...")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Test 4: Try importing new files
    new_imports = [
        ('models.plate_enhancer', 'PlateImageEnhancer'),
        ('models.enhanced_license_plate', 'SuperEnhancedLicensePlateReader'),
    ]
    
    print("\nTesting new imports...")
    for module_name, class_name in new_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
    
    return True

def test_models_init():
    """Test the models __init__.py imports."""
    print("\nTesting models.__init__.py imports...")
    print("=" * 40)
    
    try:
        from models import YOLODetector
        print("✅ YOLODetector")
    except Exception as e:
        print(f"❌ YOLODetector: {e}")
    
    try:
        from models import StableTracker
        print("✅ StableTracker")
    except Exception as e:
        print(f"❌ StableTracker: {e}")
    
    try:
        from models import TrafficLightAnalyzer
        print("✅ TrafficLightAnalyzer")
    except Exception as e:
        print(f"❌ TrafficLightAnalyzer: {e}")
    
    try:
        from models import EnhancedLicensePlateReader
        print("✅ EnhancedLicensePlateReader")
    except Exception as e:
        print(f"❌ EnhancedLicensePlateReader: {e}")
    
    try:
        from models import PlateImageEnhancer
        print("✅ PlateImageEnhancer")
    except Exception as e:
        print(f"❌ PlateImageEnhancer: {e}")
    
    try:
        from models import SuperEnhancedLicensePlateReader
        print("✅ SuperEnhancedLicensePlateReader")
    except Exception as e:
        print(f"❌ SuperEnhancedLicensePlateReader: {e}")

def check_license_plate_file():
    """Check the license_plate.py file content."""
    print("\nChecking license_plate.py file...")
    print("=" * 40)
    
    import os
    license_plate_file = 'models/license_plate.py'
    
    if not os.path.exists(license_plate_file):
        print(f"❌ File not found: {license_plate_file}")
        return
    
    try:
        with open(license_plate_file, 'r') as f:
            content = f.read()
            
        # Check for class definition
        if 'class EnhancedLicensePlateReader' in content:
            print("✅ EnhancedLicensePlateReader class found in file")
        else:
            print("❌ EnhancedLicensePlateReader class NOT found in file")
            
        # Check for imports
        if 'import easyocr' in content:
            print("✅ easyocr import found")
        else:
            print("❌ easyocr import missing")
            
        # Show first few lines
        lines = content.split('\n')[:10]
        print("\nFirst 10 lines of file:")
        for i, line in enumerate(lines, 1):
            print(f"{i:2}: {line}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    print("Debug Import Testing")
    print("=" * 50)
    
    # Step 1: Test individual imports
    test_individual_imports()
    
    # Step 2: Check license plate file
    check_license_plate_file()
    
    # Step 3: Test models init
    test_models_init()
    
    print("\n" + "=" * 50)
    print("Debug complete. Check the results above to identify the issue.")