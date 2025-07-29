"""
Command-line tool for processing violation images.

Usage:
    python process_violations.py --pending
    python process_violations.py --violation-id 123
    python process_violations.py --directory /path/to/violations
"""
import argparse
from violation_processor import ViolationProcessor

def main():
    parser = argparse.ArgumentParser(description='Process violation images for license plates')
    
    parser.add_argument('--pending', action='store_true',
                       help='Process all pending violations in database')
    parser.add_argument('--violation-id', type=int,
                       help='Process specific violation by ID')
    parser.add_argument('--directory', type=str,
                       help='Process all images in directory')
    parser.add_argument('--db', type=str, default='traffic_violations.db',
                       help='Database path')
    
    args = parser.parse_args()
    
    processor = ViolationProcessor(db_path=args.db)
    
    if args.pending:
        processed, successful = processor.process_pending_violations()
        print(f"Processed {processed} violations, {successful} successful")
        
    elif args.violation_id:
        success = processor.reprocess_violation(args.violation_id)
        if success:
            print(f"Successfully processed violation {args.violation_id}")
        else:
            print(f"Failed to process violation {args.violation_id}")
            
    elif args.directory:
        results = processor.process_directory(args.directory)
        print(f"Processed {len(results)} images")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()