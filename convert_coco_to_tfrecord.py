#!/usr/bin/env python3
"""
Convert COCO keypoint dataset to TFRecord format for faster training.
This trades disk space for significantly faster data loading.
"""

import os
import argparse
import numpy as np
from coco_cache import COCOTFRecordWriter
from data import get_cocokp_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO keypoints to TFRecord format')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (height width)')
    parser.add_argument('--samples-per-file', type=int, default=1000,
                       help='Number of samples per TFRecord file')
    parser.add_argument('--include-visibility', action='store_true',
                       help='Include visibility flags in labels')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to convert (for testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for TFRecord files')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading COCO keypoint dataset...")
    image_paths, labels = get_cocokp_paths(
        max_samples=args.max_samples,
        include_visibility=args.include_visibility,
        use_cache=True  # Use caching for faster loading
    )
    
    print(f"Loaded {len(image_paths)} samples")
    print(f"Label shape: {labels.shape}")
    
    # Create TFRecord writer
    writer = COCOTFRecordWriter(output_dir=args.output_dir)
    
    # Convert to TFRecord
    metadata = writer.convert_to_tfrecord(
        image_paths=image_paths,
        labels=labels,
        target_size=tuple(args.target_size),
        samples_per_file=args.samples_per_file,
        include_visibility=args.include_visibility
    )
    
    print("\nConversion complete!")
    print(f"Created {metadata['num_files']} TFRecord files")
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Output directory: {writer.output_dir}")
    
    # Print disk usage estimate
    tfrecord_size_gb = (metadata['num_samples'] * args.target_size[0] * args.target_size[1] * 3) / (1024**3)
    print(f"\nEstimated disk usage: ~{tfrecord_size_gb:.1f} GB")
    print("\nTo use TFRecords in training, the optimized pipeline will automatically detect and use them.")


if __name__ == '__main__':
    main()