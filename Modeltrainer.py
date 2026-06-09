import os
import torch
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class RobustPlantPredictor:
    def __init__(self, index_file='dino_centroids_cleaned.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index_file = index_file
        
        print("กำลังโหลดโมเดล DINOv2 (Robust Backbone)...")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.folder_centroids = {}
        self.class_counts = Counter()
        
        if os.path.exists(self.index_file):
            print(f"✅ พบไฟล์ {self.index_file} กำลังโหลดฐานข้อมูลพิกัด...")
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
                self.folder_centroids = data['centroids']
                self.class_counts = data['counts']
            print(f"✅ โหลดข้อมูลสำเร็จ! พร้อมใช้งานสำหรับ {len(self.folder_centroids)} คลาส")

    def _get_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.encoder(tensor).flatten().cpu().numpy()
            return embedding
        except:
            return None

    def build_index(self, train_dir_path):
        """ สแกนโฟลเดอร์เพื่อคำนวณ Centroid (ไม่ต้องลบภาพขยะแล้ว เพราะทำมาก่อนหน้านี้แล้ว) """
        if self.folder_centroids:
            print("ข้อมูล Centroid ถูกโหลดเรียบร้อยแล้ว ข้ามขั้นตอนการสร้างใหม่เพื่อประหยัดเวลา...")
            return

        valid_folders = [f for f in sorted(os.listdir(train_dir_path)) if os.path.isdir(os.path.join(train_dir_path, f))]
        total_folders = len(valid_folders)
        
        print(f"🚀 เริ่มต้นหาจุดกึ่งกลาง (Centroid) ทั้งหมด {total_folders} คลาส ใน {train_dir_path}...")
        
        for idx, folder_name in enumerate(valid_folders, 1):
            folder_path = os.path.join(train_dir_path, folder_name)
            
            # ปริ้นท์บอกก่อนเริ่มสแกนโฟลเดอร์นั้นๆ
            print(f"⏳ [{idx}/{total_folders}] กำลังสกัดจุดเด่นคลาส: {folder_name} ... ", end="", flush=True)
            
            raw_embeddings = []
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                emb = self._get_embedding(img_path)
                if emb is not None:
                    raw_embeddings.append(emb)
            
            if not raw_embeddings:
                print("❌ ข้าม (ไม่มีรูปภาพอ่านได้)")
                continue
                
            raw_embeddings = np.array(raw_embeddings)
            
            # ข้อมูลสะอาดอยู่แล้ว ใช้ Mean หรือ Median ได้เลยตรงๆ
            self.folder_centroids[folder_name] = np.median(raw_embeddings, axis=0)
            self.class_counts[folder_name] = len(raw_embeddings)
            
            # ปริ้นท์บอกว่าเสร็จแล้ว
            print(f"✅ เสร็จสิ้น ({len(raw_embeddings)} รูป)")
            
        print("✅ สร้างฐานข้อมูล Centroid สำเร็จ!")
        
        # เซฟลงไฟล์ .pkl จะได้ไม่ต้องทำใหม่คราวหน้า
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'centroids': self.folder_centroids,
                'counts': self.class_counts
            }, f)
        print(f"💾 บันทึกฐานข้อมูลลงในไฟล์ {self.index_file} เรียบร้อยแล้ว!")

    def predict(self, target_img_path, top_k=5):
        target_emb = self._get_embedding(target_img_path)
        if target_emb is None:
            return []

        results = []
        for folder_name, centroid_emb in self.folder_centroids.items():
            sim = cosine_similarity(target_emb.reshape(1, -1), centroid_emb.reshape(1, -1))[0][0]
            similarity_percentage = max(0.0, (sim + 1) / 2 * 100) 
            results.append((folder_name, similarity_percentage))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    predictor = RobustPlantPredictor()
    
    # ชี้ไปที่โฟลเดอร์ใหม่ที่ข้อมูลสะอาดแล้ว (data_cleaned)
    BASE_DATA_DIR = "processed_data/data_cleaned"
    TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"⚠️ ไม่พบโฟลเดอร์ {TRAIN_DIR} (กรุณารัน clean_and_prepare_data.py ก่อน)")
    else:
        # ไม่ต้องใส่ threshold เพื่อ drop ภาพแล้ว เพราะเราใช้เฉพาะรูปสะอาด
        predictor.build_index(TRAIN_DIR)
        
        # ทดสอบทำนายภาพแรกใน validation
        SAMPLE_VAL_IMAGE = None
        val_dir = os.path.join(BASE_DATA_DIR, "validation")
        if os.path.exists(val_dir):
            for root, dirs, files in os.walk(val_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        SAMPLE_VAL_IMAGE = os.path.join(root, file)
                        break
                if SAMPLE_VAL_IMAGE:
                    break
                    
        if SAMPLE_VAL_IMAGE and os.path.exists(SAMPLE_VAL_IMAGE):
            print(f"\n🔍 กำลังทดสอบทำนายรูป: {SAMPLE_VAL_IMAGE}")
            predictions = predictor.predict(SAMPLE_VAL_IMAGE, top_k=5)
            print(f"\n[ผลการทำนายด้วย DINOv2]")
            for folder, score in predictions:
                print(f"โฟลเดอร์: {folder} -> ความใกล้เคียง: {score:.2f}%")