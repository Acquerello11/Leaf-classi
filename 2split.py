import os
import shutil
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class AutoSynchronizedGrouper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"กำลังโหลดระบบวิเคราะห์โครงสร้างเวกเตอร์ (DINOv2) บน: {self.device}")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.encoder(tensor).flatten().cpu().numpy()
        except:
            return None

    def find_optimal_k(self, embeddings, min_k=3, max_k=15):
        """ อัลกอริทึมค้นหาจำนวนกลุ่มที่เหมาะสมที่สุดอัตโนมัติ ด้วย Silhouette Score """
        print(f"\n[AI Analysis] กำลังคำนวณหาจำนวนกลุ่มย่อยที่เหมาะสมที่สุด (จาก {min_k} ถึง {max_k} กลุ่ม)...")
        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            print(f" -> ทดสอบแบ่ง {k} กลุ่ม | คะแนนความชัดเจน (Silhouette): {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                
        print(f"*** สรุป: ระบบตัดสินใจเลือกแบ่งเป็น {best_k} กลุ่มใหญ่ (คะแนนสูงสุด: {best_score:.4f}) ***\n")
        return best_k

    def process_hierarchical_splits(self, base_data_dir, output_dir, min_groups=3, max_groups=15):
        train_dir = os.path.join(base_data_dir, "train")
        splits = ["train", "test", "validation"]
        
        print("-> ขั้นตอนที่ 1: สกัดเวกเตอร์โครงสร้างใบไม้จาก 61 คลาส (โหมด Batch Processing ความเร็วสูง)...")
        folder_names = []
        folder_embeddings = []

        dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        # โหลดภาพทีละ 128 ภาพเพื่อรีดพลังการ์ดจอ (num_workers=0 ปลอดภัยบน Windows)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        all_embs = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader, desc="สกัดเวกเตอร์ด้วย DINOv2"):
                imgs = imgs.to(self.device)
                embs = self.encoder(imgs).flatten(1).cpu().numpy()
                all_embs.append(embs)
                all_labels.append(lbls.numpy())
                
        all_embs = np.concatenate(all_embs)
        all_labels = np.concatenate(all_labels)

        print("กำลังคำนวณค่ามัธยฐาน (Median) ประจำสายพันธุ์...")
        for idx, cls_name in enumerate(dataset.classes):
            cls_embs = all_embs[all_labels == idx]
            if len(cls_embs) > 0:
                folder_embeddings.append(np.median(cls_embs, axis=0))
                folder_names.append(cls_name)

        if not folder_embeddings:
            print("เกิดข้อผิดพลาด: ไม่พบข้อมูลรูปภาพเลย")
            return

        # แปลงเป็น NumPy Array เพื่อเข้าอัลกอริทึมคณิตศาสตร์
        folder_embeddings = np.array(folder_embeddings)

        # 2. ให้ AI หาจำนวนกลุ่มที่ดีที่สุด (Auto K-Means)
        optimal_k = self.find_optimal_k(folder_embeddings, min_k=min_groups, max_k=max_groups)
        
        print("-> ขั้นตอนที่ 2: ทำการจับกลุ่มคลาสพืชตามจำนวนที่ AI เลือก...")
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(folder_embeddings)
        
        class_to_group_map = dict(zip(folder_names, cluster_labels))

        # 2.5 สร้างรายงานผลและกราฟ PCA Scatter Plot
        print("-> สร้างรายงานการจัดกลุ่ม (Clustering Report) และกราฟ PCA...")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "clustering_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== รายงานสรุปการแบ่งกลุ่ม Super Class ({optimal_k} กลุ่ม) ===\n")
            f.write("จุดประสงค์: อธิบายว่าสายพันธุ์ไหนถูกจัดให้อยู่ด้วยกันเพราะ AI มองเห็นโครงสร้างใบที่คล้ายกัน\n\n")
            for i in range(optimal_k):
                members = [name for name, group in class_to_group_map.items() if group == i]
                f.write(f"Super Group {i} ({len(members)} สายพันธุ์):\n")
                for member in members:
                    f.write(f"  - {member}\n")
                f.write("\n")
                
        # สร้างกราฟ PCA (ลดมิติจาก 384 หรือ 768 เหลือ 2 มิติเพื่อวาดกราฟ)
        pca = PCA(n_components=2, random_state=42)
        reduced_embs = pca.fit_transform(folder_embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced_embs[:, 0], y=reduced_embs[:, 1], hue=cluster_labels, palette="tab10", s=100, alpha=0.8)
        for i, name in enumerate(folder_names):
            plt.text(reduced_embs[i, 0] + 0.02, reduced_embs[i, 1] + 0.02, name, fontsize=6, alpha=0.7)
        plt.title(f"DINOv2 Embeddings PCA Scatter Plot ({optimal_k} Clusters)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Super Group")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cluster_pca_plot.png"), dpi=300)
        plt.close()

        # 3. กระจายโครงสร้างโฟลเดอร์ไปยัง Train, Test, Validate
        print("-> ขั้นตอนที่ 3: กำลังคัดแยกโฟลเดอร์สปลิตทั้งหมดลงกลุ่มโครงสร้างใหม่...")
        for split in tqdm(splits, desc="กำลังคัดแยกไฟล์เข้าโฟลเดอร์"):
            source_split_dir = os.path.join(base_data_dir, split)
            if not os.path.exists(source_split_dir):
                continue
            for folder_name in os.listdir(source_split_dir):
                if folder_name not in class_to_group_map:
                    continue
                
                group_id = class_to_group_map[folder_name]
                
                target_dir = os.path.join(output_dir, f"super_group_{group_id}", split, folder_name)
                source_dir = os.path.join(source_split_dir, folder_name)
                
                if not os.path.exists(target_group_dir := os.path.dirname(target_dir)):
                    os.makedirs(target_group_dir, exist_ok=True)
                
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                
            print(f"    [Success] จัดหมวดหมู่เซต {split} สำเร็จ")

        print(f"\n[เสร็จสมบูรณ์] จัดโครงสร้างได้ทั้งหมด {optimal_k} กลุ่ม พร้อมใช้งานที่พาธ: {output_dir}")

if __name__ == "__main__":
    # ปรับพิกัดพาธให้ตรงตามโครงสร้างของคุณ
    SOURCE_DIR = "processed_data/data_cleaned"
    TARGET_DIR = "processed_data/hierarchical_splits"
    
    grouper = AutoSynchronizedGrouper()
    # กำหนดช่วงให้ AI ลองหาจำนวนกลุ่มที่เหมาะสมที่สุดระหว่าง 12 ถึง 25 กลุ่ม (แบ่งละเอียดขึ้น)
    grouper.process_hierarchical_splits(SOURCE_DIR, TARGET_DIR, min_groups=12, max_groups=25)