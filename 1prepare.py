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
    def __init__(self, source_dir, dest_dir):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        
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
                print(f"⏩ ข้ามคลาส '{cls}' เพราะเคยประมวลผลเสร็จแล้ว (Resume)")
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
            
            # 2. จัดเรียงรูปภาพตามคุณภาพ (ความเหมือนกับค่าเฉลี่ยของกลุ่ม)
            if total_images_in_class > 4:
                median_emb = np.median(raw_embeddings, axis=0)
                sims = cosine_similarity(raw_embeddings, median_emb.reshape(1, -1)).flatten()
                
                # จับคู่ชื่อไฟล์กับคะแนนความเหมือน
                scored_images = list(zip(valid_image_names, sims))
                # เรียงลำดับจากคะแนนมาก (ภาพใช้งานได้ดี) ไปน้อย (ภาพแปลกปลอม/ใช้งานไม่ได้)
                scored_images.sort(key=lambda x: x[1], reverse=True)
                sorted_image_names = [x[0] for x in scored_images]
            else:
                # ถ้าภาพน้อยเกินไป ให้สุ่มแทน
                sorted_image_names = valid_image_names.copy()
                random.shuffle(sorted_image_names)
                
            total_kept += total_images_in_class
            
            # 3. Split data: 
            # - ส่วนมากที่ใช้งานได้ -> Train (70%)
            # - อีกส่วนที่ใช้งานได้ -> Val (15%)
            # - ส่วนมากที่ใช้งานไม่ได้ (คะแนนน้อยสุดอยู่ท้ายสุด) -> Test (15%)
            n_total = total_images_in_class
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_images = sorted_image_names[:n_train]
            val_images = sorted_image_names[n_train:n_train+n_val]
            test_images = sorted_image_names[n_train+n_val:]
            
            print(f"✅ จัดการ {n_total} รูป -> แบ่งเป็น: Train={len(train_images)} (ภาพดี) | Val={len(val_images)} | Test={len(test_images)} (ภาพแย่)")
            
            # 4. Copy Files
            for img in train_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
            for img in val_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
            for img in test_images:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
                
        print("\n================================================")
        print("🎉 ประมวลผลและจัดกลุ่มรูปภาพเสร็จสมบูรณ์!")
        print(f"✅ จำนวนรูปทั้งหมดที่ถูกนำไปกระจายลง Train/Val/Test: {total_kept} รูป")
        print(f"📁 ข้อมูลทั้งหมดถูกบันทึกไว้ที่: {self.dest_dir}")
        print("================================================")

if __name__ == "__main__":
    # ต้นทางคือโฟลเดอร์รูปภาพออริจินัลทั้งหมด
    SOURCE_DIR = 'Data200'
    # ปลายทางเป็นโฟลเดอร์ใหม่ เพื่อไม่ให้ปนกับของเก่า
    DEST_DIR = 'processed_data/data_cleaned'
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory {SOURCE_DIR} not found.")
    else:
        preparer = DataCleanerAndPreparer(SOURCE_DIR, DEST_DIR)
        preparer.process_and_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
