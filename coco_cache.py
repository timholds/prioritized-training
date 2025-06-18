"""
COCO dataset caching utilities for faster training startup.
Provides caching for annotations, image paths, and IL losses.
"""

import os
import json
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from tqdm import tqdm


class COCOCache:
    """Handles caching of COCO annotations, image paths, and IL losses."""
    
    def __init__(self, base_dir: str = 'datasets/coco2017', cache_dir: str = '.coco_cache'):
        self.base_dir = base_dir
        self.cache_dir = os.path.join(base_dir, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, cache_type: str, suffix: str = '') -> str:
        """Get path for a specific cache file."""
        filename = f"{cache_type}{suffix}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for cache validation."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def load_cached_annotations(self) -> Dict[str, Any]:
        """Load cached annotations or parse from JSON files."""
        cache_path = self._get_cache_path('annotations')
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading cached annotations from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data['annotations'])} annotations from cache")
            return data
        
        # Load from original JSON files
        print("Parsing COCO annotations (this will be cached for future runs)...")
        train_path = os.path.join(self.base_dir, 'annotations', 'person_keypoints_train2017.json')
        val_path = os.path.join(self.base_dir, 'annotations', 'person_keypoints_val2017.json')
        
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        
        # Combine annotations and create image info mapping
        all_annotations = []
        image_info = {}
        
        # Process train annotations
        for ann in train_data['annotations']:
            ann['split'] = 'train2017'
            all_annotations.append(ann)
        
        # Process val annotations
        for ann in val_data['annotations']:
            ann['split'] = 'val2017'
            all_annotations.append(ann)
        
        # Create image info mapping
        for img in train_data['images']:
            img['split'] = 'train2017'
            image_info[img['id']] = img
        for img in val_data['images']:
            img['split'] = 'val2017'
            image_info[img['id']] = img
        
        # Cache the combined data
        data = {
            'annotations': all_annotations,
            'image_info': image_info,
            'train_images': len(train_data['images']),
            'val_images': len(val_data['images'])
        }
        
        print(f"Caching {len(all_annotations)} annotations...")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def _validate_image_path(self, img_path: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """Check if image exists and get its dimensions."""
        if os.path.exists(img_path):
            try:
                # Use PIL to get image dimensions without loading full image
                from PIL import Image
                with Image.open(img_path) as img:
                    return True, img.width, img.height
            except:
                return False, None, None
        return False, None, None
    
    def build_image_path_index(self, force_rebuild: bool = False) -> Dict[int, Dict[str, Any]]:
        """Build or load cached index of image paths with validation."""
        cache_path = self._get_cache_path('image_index')
        
        if not force_rebuild and os.path.exists(cache_path):
            print(f"Loading cached image index from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print("Building image path index with validation (this will be cached)...")
        ann_data = self.load_cached_annotations()
        image_info = ann_data['image_info']
        
        image_index = {}
        
        # Use thread pool for parallel path validation
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all validation tasks
            future_to_img_id = {}
            for img_id, img_info in image_info.items():
                img_path = os.path.join(self.base_dir, img_info['split'], img_info['file_name'])
                future = executor.submit(self._validate_image_path, img_path)
                future_to_img_id[future] = (img_id, img_path, img_info)
            
            # Process results with progress bar
            for future in tqdm(as_completed(future_to_img_id), total=len(future_to_img_id), 
                             desc="Validating image paths"):
                img_id, img_path, img_info = future_to_img_id[future]
                exists, width, height = future.result()
                
                image_index[img_id] = {
                    'path': img_path,
                    'exists': exists,
                    'width': width or img_info.get('width', 0),
                    'height': height or img_info.get('height', 0),
                    'split': img_info['split'],
                    'file_name': img_info['file_name']
                }
        
        # Count valid images
        valid_count = sum(1 for info in image_index.values() if info['exists'])
        print(f"Found {valid_count}/{len(image_index)} valid images")
        
        # Cache the index
        with open(cache_path, 'wb') as f:
            pickle.dump(image_index, f)
        
        return image_index
    
    def get_il_loss_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for IL losses based on model configuration."""
        key_parts = [
            config.get('dataset', 'cocokp'),
            config.get('holdout_model', config['model']).__name__,
            str(config.get('holdout_epochs', 10)),
            config.get('loss', 'mse'),
            str(config.get('input_shape', (224, 224, 3))),
            str(config.get('include_visibility', False))
        ]
        return '_'.join(key_parts)
    
    def load_cached_il_losses(self, config: Dict[str, Any]) -> Optional[Dict[int, float]]:
        """Load cached IL losses if available."""
        cache_key = self.get_il_loss_cache_key(config)
        cache_path = self._get_cache_path('il_losses', f'_{cache_key}')
        
        if os.path.exists(cache_path):
            print(f"Loading cached IL losses from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def cache_il_losses(self, config: Dict[str, Any], il_loss_dict: Dict[int, float]):
        """Cache IL losses for future use."""
        cache_key = self.get_il_loss_cache_key(config)
        cache_path = self._get_cache_path('il_losses', f'_{cache_key}')
        
        print(f"Caching IL losses to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(il_loss_dict, f)


class COCOTFRecordWriter:
    """Convert COCO dataset to TFRecord format for faster loading."""
    
    def __init__(self, base_dir: str = 'datasets/coco2017', output_dir: str = None):
        self.base_dir = base_dir
        self.output_dir = output_dir or os.path.join(base_dir, 'tfrecords')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_list_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def convert_to_tfrecord(self, 
                           image_paths: List[str], 
                           labels: np.ndarray,
                           target_size: Tuple[int, int] = (224, 224),
                           samples_per_file: int = 1000,
                           include_visibility: bool = False):
        """Convert COCO keypoint data to TFRecord format."""
        
        print(f"Converting {len(image_paths)} samples to TFRecord format...")
        print(f"Target size: {target_size}")
        print(f"Samples per file: {samples_per_file}")
        print(f"Output directory: {self.output_dir}")
        
        num_files = (len(image_paths) + samples_per_file - 1) // samples_per_file
        
        for file_idx in range(num_files):
            start_idx = file_idx * samples_per_file
            end_idx = min(start_idx + samples_per_file, len(image_paths))
            
            tfrecord_path = os.path.join(self.output_dir, f'coco_keypoints_{file_idx:04d}.tfrecord')
            
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for i in tqdm(range(start_idx, end_idx), 
                            desc=f"Writing file {file_idx + 1}/{num_files}"):
                    
                    try:
                        # Load and preprocess image
                        from PIL import Image
                        img = Image.open(image_paths[i]).convert('RGB')
                        orig_w, orig_h = img.size
                        img = img.resize(target_size)
                        img_array = np.array(img, dtype=np.uint8)
                        
                        # Encode as JPEG
                        img_encoded = tf.io.encode_jpeg(img_array).numpy()
                        
                        # Create feature dictionary
                        feature = {
                            'image': self._bytes_feature(img_encoded),
                            'label': self._float_list_feature(labels[i]),
                            'height': self._int64_feature(target_size[1]),
                            'width': self._int64_feature(target_size[0]),
                            'orig_height': self._int64_feature(orig_h),
                            'orig_width': self._int64_feature(orig_w),
                        }
                        
                        # Create Example and write
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                        
                    except Exception as e:
                        print(f"Error processing {image_paths[i]}: {e}")
                        continue
            
            print(f"Written {tfrecord_path}")
        
        # Write metadata
        metadata = {
            'num_samples': len(image_paths),
            'num_files': num_files,
            'samples_per_file': samples_per_file,
            'target_size': target_size,
            'include_visibility': include_visibility,
            'label_shape': labels.shape[1] if len(labels.shape) > 1 else 1
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Conversion complete! Metadata saved to {metadata_path}")
        return metadata