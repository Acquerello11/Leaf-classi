import os
import shutil
import random

def prepare_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Set seed for reproducibility
    random.seed(42)
    
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'validation')
    test_dir = os.path.join(dest_dir, 'test')
    
    # Create directories
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Filter classes to keep only original folders, excluding those containing "edited" or "Edited"
    valid_classes = [c for c in classes if "edited" not in c.lower()]
    
    print(f"Found {len(classes)} total classes, keeping {len(valid_classes)} original classes.")
    
    for cls in valid_classes:
        cls_path = os.path.join(source_dir, cls)
        
        # Create class subdirectories in dest
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        # Get all images
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        # Calculate splits
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        print(f"Class '{cls}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        # Copy files
        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
            
        for img in val_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
            
        for img in test_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
            
    print("Data preparation complete.")

if __name__ == "__main__":
    SOURCE_DIR = 'T-Leaf(From3273080)/Data200'
    DEST_DIR = 'processed_data/data'
    
    # Ensure source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory {SOURCE_DIR} not found.")
    else:
        prepare_data(SOURCE_DIR, DEST_DIR, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
