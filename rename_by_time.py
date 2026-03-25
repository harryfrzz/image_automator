#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def rename_images(folder):
    """Rename all images in subfolders to DWP_1, DWP_2, etc."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # Get all subdirectories
    subdirs = [d for d in Path(folder).iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories")
    
    total_renamed = 0
    
    for subdir in sorted(subdirs):
        images = []
        for ext in image_extensions:
            images.extend(subdir.glob(f'*{ext}'))
            images.extend(subdir.glob(f'*{ext.upper()}'))
        
        print(f"\n{subdir.name}: {len(images)} images")
        
        counter = 1
        renamed = 0
        
        for img_path in sorted(images):
            ext = img_path.suffix
            new_name = f"DWP_{counter}{ext}"
            new_path = img_path.parent / new_name
            
            # Handle duplicates
            if new_path.exists():
                c = 1
                while new_path.exists():
                    new_name = f"DWP_{counter}_{c}{ext}"
                    new_path = img_path.parent / new_name
                    c += 1
            
            os.rename(img_path, new_path)
            print(f"  {new_name}")
            renamed += 1
            counter += 1
        
        total_renamed += renamed
    
    print(f"\nDone! Total renamed: {total_renamed}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "output_images"
    rename_images(folder)
