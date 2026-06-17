import os
import shutil
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ImagePathDataset(datasets.ImageFolder):
    """ Custom dataset that includes the original image paths. """
    def __getitem__(self, index):
        original_tuple = super(ImagePathDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], original_tuple[1], path

class UnifiedDataProcessor:
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

    def find_optimal_k(self, embeddings, min_k=12, max_k=25):
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

    def process(self, source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_groups=12, max_groups=25):
        random.seed(42)
        np.random.seed(42)

        # 1. โหลดข้อมูลภาพทั้งหมดในโหมด Batch
        print("-> ขั้นตอนที่ 1: สกัดเวกเตอร์โครงสร้างใบไม้จากทุกคลาส (Batch Processing)...")
        dataset = ImagePathDataset(source_dir, transform=self.transform)
        # num_workers=0 ปลอดภัยสำหรับ Windows
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        all_embs = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for imgs, lbls, paths in tqdm(dataloader, desc="สกัดเวกเตอร์ด้วย DINOv2"):
                imgs = imgs.to(self.device)
                embs = self.encoder(imgs).flatten(1).cpu().numpy()
                all_embs.append(embs)
                all_labels.append(lbls.numpy())
                all_paths.extend(paths)

        all_embs = np.concatenate(all_embs)
        all_labels = np.concatenate(all_labels)
        all_paths = np.array(all_paths)

        print("\n-> ขั้นตอนที่ 2: ตรวจสอบคุณภาพและแบ่งชุดข้อมูล (Train/Val/Test)...")
        class_splits = {}
        class_median_embs = []
        valid_class_names = []
        total_kept = 0

        for idx, cls_name in enumerate(dataset.classes):
            if "edited" in cls_name.lower():
                continue
                
            mask = (all_labels == idx)
            cls_embs = all_embs[mask]
            cls_paths = all_paths[mask]
            
            if len(cls_embs) == 0:
                continue
                
            n_total = len(cls_paths)
            
            # จัดเรียงภาพตามคุณภาพความคล้ายคลึงกับค่ามัธยฐานของกลุ่มตัวเอง
            if n_total > 4:
                median_emb = np.median(cls_embs, axis=0)
                sims = cosine_similarity(cls_embs, median_emb.reshape(1, -1)).flatten()
                sorted_indices = np.argsort(sims)[::-1] # เรียงจากมากไปน้อย
            else:
                sorted_indices = np.random.permutation(n_total)
                
            sorted_paths = cls_paths[sorted_indices]
            
            # แบ่งสัดส่วน Train, Val, Test
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_paths = sorted_paths[:n_train]
            val_paths = sorted_paths[n_train:n_train+n_val]
            test_paths = sorted_paths[n_train+n_val:]
            
            class_splits[cls_name] = {
                "train": train_paths,
                "validation": val_paths,
                "test": test_paths
            }
            
            print(f"✅ คลาส '{cls_name}' ({n_total} รูป) -> Train={len(train_paths)} | Val={len(val_paths)} | Test={len(test_paths)}")
            total_kept += n_total
            
            # สกัดหาจุดศูนย์กลางที่แม่นยำขึ้นจากเฉพาะข้อมูล Train (หลีกเลี่ยงภาพขยะใน Test)
            train_mask = np.isin(cls_paths, train_paths)
            train_embs = cls_embs[train_mask]
            
            if len(train_embs) > 0:
                class_median_embs.append(np.median(train_embs, axis=0))
            else:
                class_median_embs.append(np.median(cls_embs, axis=0))
                
            valid_class_names.append(cls_name)

        if not class_median_embs:
            print("เกิดข้อผิดพลาด: ไม่พบข้อมูลรูปภาพที่นำไปใช้งานได้")
            return

        class_median_embs = np.array(class_median_embs)

        # 3. วิเคราะห์จัดกลุ่ม Super Groups ตามโครงสร้าง (K-Means)
        optimal_k = self.find_optimal_k(class_median_embs, min_k=min_groups, max_k=max_groups)
        
        print("-> ขั้นตอนที่ 3: ดำเนินการจับกลุ่มคลาสพืชตามจำนวนที่ AI เลือก...")
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(class_median_embs)
        
        class_to_group_map = dict(zip(valid_class_names, cluster_labels))

        # สร้างรายงานและกราฟ PCA
        os.makedirs(target_dir, exist_ok=True)
        report_path = os.path.join(target_dir, "clustering_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== รายงานสรุปการแบ่งกลุ่ม Super Class ({optimal_k} กลุ่ม) ===\n")
            f.write("จุดประสงค์: อธิบายว่าสายพันธุ์ไหนถูกจัดให้อยู่ด้วยกันเพราะ AI มองเห็นโครงสร้างใบที่คล้ายกัน\n\n")
            for i in range(optimal_k):
                members = [name for name, group in class_to_group_map.items() if group == i]
                f.write(f"Super Group {i} ({len(members)} สายพันธุ์):\n")
                for member in members:
                    f.write(f"  - {member}\n")
                f.write("\n")
                
        pca = PCA(n_components=2, random_state=42)
        reduced_embs = pca.fit_transform(class_median_embs)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced_embs[:, 0], y=reduced_embs[:, 1], hue=cluster_labels, palette="tab10", s=100, alpha=0.8)
        for i, name in enumerate(valid_class_names):
            plt.text(reduced_embs[i, 0] + 0.02, reduced_embs[i, 1] + 0.02, name, fontsize=6, alpha=0.7)
        plt.title(f"DINOv2 Embeddings PCA Scatter Plot ({optimal_k} Clusters)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Super Group")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "cluster_pca_plot.png"), dpi=300)
        plt.close()

        # 4. คัดลอกรูปภาพลงโฟลเดอร์ Hierarchical
        print("-> ขั้นตอนที่ 4: กระจายรูปภาพลงโฟลเดอร์ Hierarchical Splits โดยตรง (ประหยัดพื้นที่จัดเก็บ)...")
        
        for cls_name, splits in tqdm(class_splits.items(), desc="คัดลอกไฟล์"):
            group_id = class_to_group_map[cls_name]
            group_dir = os.path.join(target_dir, f"super_group_{group_id}")
            
            for split_name, paths in splits.items():
                target_class_dir = os.path.join(group_dir, split_name, cls_name)
                
                if os.path.exists(target_class_dir):
                    shutil.rmtree(target_class_dir)
                os.makedirs(target_class_dir, exist_ok=True)
                
                for p in paths:
                    # คัดลอกไฟล์ไปยังปลายทาง
                    shutil.copy2(p, os.path.join(target_class_dir, os.path.basename(p)))
                    
        print("\n================================================")
        print("🎉 ประมวลผล ควบคุมคุณภาพ แบ่งชุดฝึก และจัดกลุ่มรูปภาพเสร็จสมบูรณ์!")
        print(f"✅ จำนวนรูปทั้งหมดที่ถูกจัดการ: {total_kept} รูป")
        print(f"📁 โครงสร้างจัดเก็บใหม่: {target_dir}")
        print("================================================")

if __name__ == "__main__":
    SOURCE_DIR = 'Data200'
    TARGET_DIR = 'processed_data/hierarchical_splits'
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
    else:
        processor = UnifiedDataProcessor()
        processor.process(SOURCE_DIR, TARGET_DIR, min_groups=12, max_groups=25)
