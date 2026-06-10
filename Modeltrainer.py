import os
import torch
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
from sklearn.svm import SVC
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
        
        self.classifier = None
        
        if os.path.exists(self.index_file):
            print(f"✅ พบไฟล์ {self.index_file} กำลังโหลดฐานข้อมูลโมเดล SVM...")
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
                self.classifier = data.get('classifier')
            if self.classifier is not None:
                print(f"✅ โหลดข้อมูลสำเร็จ! พร้อมใช้งานสำหรับ {len(self.classifier.classes_)} คลาส")

    def _get_embedding(self, img_input):
        try:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert('RGB')
            else:
                img = img_input.convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.encoder(tensor).flatten().cpu().numpy()
            return embedding
        except:
            return None

    def build_index(self, train_dir_path):
        """ สแกนโฟลเดอร์เพื่อสกัด Features และสอนโมเดล SVM """
        # We don't skip here if we want to force retrain, but let's keep the check
        # Actually, let's remove the skip if we want to retrain because the old file might just have centroids.
        # It's better to force a retrain if we are calling this.
        
        valid_folders = [f for f in sorted(os.listdir(train_dir_path)) if os.path.isdir(os.path.join(train_dir_path, f))]
        total_folders = len(valid_folders)
        
        print(f"🚀 เริ่มต้นสกัดจุดเด่น (Features) ทั้งหมด {total_folders} คลาส ใน {train_dir_path}...")
        
        X = []
        y = []
        
        for idx, folder_name in enumerate(valid_folders, 1):
            folder_path = os.path.join(train_dir_path, folder_name)
            
            print(f"⏳ [{idx}/{total_folders}] กำลังสกัดจุดเด่นคลาส: {folder_name} ... ", end="", flush=True)
            
            count = 0
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                emb = self._get_embedding(img_path)
                if emb is not None:
                    X.append(emb)
                    y.append(folder_name)
                    count += 1
            
            if count == 0:
                print("❌ ข้าม (ไม่มีรูปภาพอ่านได้)")
            else:
                print(f"✅ เสร็จสิ้น ({count} รูป)")
            
        print("✅ สกัด Features สำเร็จ! เริ่มฝึกสอนโมเดล SVM...")
        self.classifier = SVC(kernel='linear', probability=True, class_weight='balanced')
        self.classifier.fit(X, y)
        print("✅ ฝึกสอนโมเดล SVM สำเร็จ!")
        
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier
            }, f)
        print(f"💾 บันทึกโมเดลลงในไฟล์ {self.index_file} เรียบร้อยแล้ว!")

    def predict(self, target_img_input, top_k=5):
        target_emb = self._get_embedding(target_img_input)
        if target_emb is None or self.classifier is None:
            return []

        probs = self.classifier.predict_proba(target_emb.reshape(1, -1))[0]
        classes = self.classifier.classes_
        
        results = [(cls, prob * 100.0) for cls, prob in zip(classes, probs)]
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