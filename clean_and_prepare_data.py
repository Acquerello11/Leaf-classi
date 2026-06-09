import os
import shutil
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class DataCleanerAndPreparer:
    def __init__(self, source_dir, dest_dir, drop_threshold=1.5):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.drop_threshold = drop_threshold
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("กำลังโหลดโมเดล DINOv2 (เพื่อทำหน้าที่ QC คัดแยกรูปภาพขยะ)...")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def _get_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.encoder(tensor).flatten().cpu().numpy()
            return embedding
        except:
            return None

    def process_and_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        random.seed(42)
        
        train_dir = os.path.join(self.dest_dir, 'train')
        val_dir = os.path.join(self.dest_dir, 'validation')
        test_dir = os.path.join(self.dest_dir, 'test')
        
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)
            
        classes = [d for d in os.listdir(self.source_dir) if os.path.isdir(os.path.join(self.source_dir, d))]
        valid_classes = [c for c in classes if "edited" not in c.lower()]
        
        print(f"พบข้อมูลทั้งหมด {len(classes)} คลาส (เลือกใช้เฉพาะออริจินัล {len(valid_classes)} คลาส)")
        
        total_kept = 0
        total_dropped = 0
        
        for cls in valid_classes:
            cls_path = os.path.join(self.source_dir, cls)
            
            # --- ระบบ Resume: เช็คว่าคลาสนี้เคยทำเสร็จแล้วหรือยัง ---
            check_train_path = os.path.join(train_dir, cls)
            check_val_path = os.path.join(val_dir, cls)
            check_test_path = os.path.join(test_dir, cls)
            
            # ถ้ามีโฟลเดอร์นี้อยู่แล้ว และมีไฟล์อยู่ข้างใน ถือว่าทำเสร็จแล้ว ข้ามได้เลย
            if os.path.exists(check_train_path) and len(os.listdir(check_train_path)) > 0:
                print(f"⏩ ข้ามคลาส '{cls}' เพราะเคยประมวลผลและคัดขยะเสร็จแล้ว (Resume)")
                # นับรวมยอดเดิมด้วยคร่าวๆ
                total_kept += len(os.listdir(check_train_path)) + len(os.listdir(check_val_path)) + len(os.listdir(check_test_path))
                continue
                
            os.makedirs(check_train_path, exist_ok=True)
            os.makedirs(check_val_path, exist_ok=True)
            os.makedirs(check_test_path, exist_ok=True)
            
            all_images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\n🔍 กำลังตรวจสอบภาพในคลาส '{cls}' ({len(all_images)} รูป)...")
            
            # 1. Extract Embeddings
            raw_embeddings = []
            valid_image_names = []
            
            for img_name in all_images:
                img_path = os.path.join(cls_path, img_name)
                emb = self._get_embedding(img_path)
                if emb is not None:
                    raw_embeddings.append(emb)
                    valid_image_names.append(img_name)
                    
            if not raw_embeddings:
                print(f"ข้ามคลาส {cls} เพราะไม่มีรูปภาพที่อ่านได้")
                continue
                
            raw_embeddings = np.array(raw_embeddings)
            total_images_in_class = len(raw_embeddings)
            
            # 2. Outlier Rejection (Drop Bad Images)
            clean_images = []
            
            if total_images_in_class > 4:
                median_emb = np.median(raw_embeddings, axis=0)
                sims = cosine_similarity(raw_embeddings, median_emb.reshape(1, -1)).flatten()
                
                mad = np.median(np.abs(sims - np.median(sims)))
                if mad == 0: mad = 1e-6
                
                robust_z_scores = 0.67449 * (sims - np.median(sims)) / mad
                valid_mask = robust_z_scores >= -self.drop_threshold
                
                for idx, is_valid in enumerate(valid_mask):
                    if is_valid:
                        clean_images.append(valid_image_names[idx])
                        
                dropped_count = total_images_in_class - len(clean_images)
                if dropped_count > 0:
                    print(f"🗑️ ลบทิ้ง {dropped_count} รูป (คุณภาพต่ำ/ฉากหลังรก/ไม่ใช่ใบไม้)")
            else:
                clean_images = valid_image_names
                
            kept_count = len(clean_images)
            total_kept += kept_count
            total_dropped += (total_images_in_class - kept_count)
            
            # 3. Split data 70/15/15
            random.shuffle(clean_images)
            n_total = len(clean_images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_images = clean_images[:n_train]
            val_images = clean_images[n_train:n_train+n_val]
            test_images = clean_images[n_train+n_val:]
            
            print(f"✅ ผ่าน QC {kept_count} รูป -> แบ่งเป็น: Train={len(train_images)} | Val={len(val_images)} | Test={len(test_images)}")
            
            # 4. Copy Files
            for img in train_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
            for img in val_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
            for img in test_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
                
        print("\n================================================")
        print("🎉 ประมวลผลและคัดกรองข้อมูลเสร็จสมบูรณ์!")
        print(f"✅ รูปที่ผ่าน QC และถูกนำไปใช้งาน: {total_kept} รูป")
        print(f"🗑️ รูปที่ถูกคัดทิ้ง (Noise/Outlier): {total_dropped} รูป")
        print(f"📁 ข้อมูลสะอาดพร้อมเทรนถูกบันทึกไว้ที่: {self.dest_dir}")
        print("================================================")

if __name__ == "__main__":
    # ต้นทางคือโฟลเดอร์รูปภาพออริจินัลทั้งหมด
    SOURCE_DIR = 'T-Leaf(From3273080)/Data200'
    # ปลายทางเป็นโฟลเดอร์ใหม่ เพื่อไม่ให้ปนกับของเก่า
    DEST_DIR = 'processed_data/data_cleaned'
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory {SOURCE_DIR} not found.")
    else:
        # drop_threshold = 1.5 คือเกณฑ์การคัดทิ้ง ถ้ายิ่งน้อย (เช่น 1.0) ยิ่งคัดทิ้งโหดมาก
        preparer = DataCleanerAndPreparer(SOURCE_DIR, DEST_DIR, drop_threshold=1.5)
        preparer.process_and_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
