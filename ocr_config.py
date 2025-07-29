"""
OCR Configuration for License Plate Recognition
Adjust these settings based on your specific license plate format and image quality.
"""

# License plate format patterns for your region
# Adjust these based on your local license plate formats
LICENSE_PLATE_PATTERNS = [
    r'^[A-Z]{3}[0-9]{3}$',      # ABC123 (3 letters, 3 numbers)
    r'^[A-Z]{2}[0-9]{4}$',      # AB1234 (2 letters, 4 numbers)
    r'^[A-Z][0-9]{3}[A-Z]{3}$', # A123ABC (1 letter, 3 numbers, 3 letters)
    r'^[0-9]{3}[A-Z]{3}$',      # 123ABC (3 numbers, 3 letters)
    r'^[A-Z]{4}[0-9]{2}$',      # ABCD12 (4 letters, 2 numbers)
    r'^[0-9]{2}[A-Z]{4}$',      # 12ABCD (2 numbers, 4 letters)
    r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$', # A12ABC (1 letter, 2 numbers, 3 letters)
    r'^[0-9]{1}[A-Z]{3}[0-9]{3}$', # 1ABC123 (1 number, 3 letters, 3 numbers)
]

# Character correction mappings (common OCR mistakes)
CHARACTER_CORRECTIONS = {
    # Position-based corrections
    'position_based': {
        'first_letters': {  # First 1-3 characters are usually letters
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '8': 'B',
            '6': 'G',
            '2': 'Z'
        },
        'last_numbers': {   # Last 2-4 characters are usually numbers
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8',
            'G': '6',
            'Z': '2'
        }
    },
    
    # Common character confusions
    'common_mistakes': {
        'D': '0',  # D often confused with 0
        'Q': '0',  # Q often confused with 0
        'U': '0',  # U often confused with 0
        'T': '7',  # T often confused with 7
        'Y': '7',  # Y often confused with 7
        'A': '4',  # A sometimes confused with 4
        'E': '3',  # E sometimes confused with 3
    }
}

# OCR Enhancement Settings
OCR_ENHANCEMENT_SETTINGS = {
    'min_plate_width': 80,      # Minimum width for plate processing
    'min_plate_height': 25,     # Minimum height for plate processing
    'upscale_factor': 2.5,      # Scale factor for small plates
    'max_plate_width': 400,     # Maximum width to prevent over-processing
    
    # Enhancement levels
    'enhancement_levels': {
        'light': {
            'denoise_strength': 5,
            'contrast_factor': 1.1,
            'sharpen_strength': 0.3
        },
        'medium': {
            'denoise_strength': 8,
            'contrast_factor': 1.3,
            'sharpen_strength': 0.6
        },
        'aggressive': {
            'denoise_strength': 12,
            'contrast_factor': 1.5,
            'sharpen_strength': 1.0
        }
    }
}

# EasyOCR Parameters
EASYOCR_PARAMS = {
    'default': {
        'width_ths': 0.7,
        'height_ths': 0.7,
        'paragraph': False,
        'detail': 1,
        'decoder': 'greedy',
        'beamWidth': 5
    },
    'sensitive': {
        'width_ths': 0.4,
        'height_ths': 0.4,
        'paragraph': False,
        'detail': 1,
        'decoder': 'greedy',
        'beamWidth': 10
    },
    'conservative': {
        'width_ths': 0.9,
        'height_ths': 0.9,
        'paragraph': False,
        'detail': 1,
        'decoder': 'greedy',
        'beamWidth': 3
    }
}

# Validation Settings
VALIDATION_SETTINGS = {
    'min_plate_length': 4,
    'max_plate_length': 8,
    'min_confidence': 0.3,
    'require_letters_and_numbers': True,
    'consensus_threshold': 0.8,  # Similarity threshold for grouping results
}

# Debug Settings
DEBUG_SETTINGS = {
    'save_debug_images': True,
    'save_all_attempts': False,  # Save all OCR attempts vs just best ones
    'debug_dir_name': 'ocr_debug',
    'enhanced_dir_name': 'enhanced_debug'
}