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
import cv2

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

        # ลงทะเบียน Forward Hook เพื่อดึงข้อมูลนำเข้า (x) ของบล็อกสุดท้ายของ DINOv2
        self.attn_input = None
        def hook_fn(module, input, output):
            self.attn_input = input[0]
        self.encoder.blocks[-1].attn.register_forward_hook(hook_fn)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.scan_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def clean_and_crop_image(self, img_path, class_name, cleaned_dir, discarded_dir, threshold=0.5, min_area_ratio=0.02):
        """
        Loads an image, runs DINOv2 self-attention to detect leaf regions,
        crops them, and saves to cleaned_dir. If no leaf is found, copies to discarded_dir.
        """
        try:
            img_org = Image.open(img_path).convert('RGB')
            w_org, h_org = img_org.size
            
            # Prepare image for DINOv2 (224x224) without CenterCrop to scan the whole image
            img_tensor = self.scan_transform(img_org).unsqueeze(0).to(self.device)
            
            self.attn_input = None
            with torch.no_grad():
                # รัน forward pass เพื่อกระตุ้น hook
                _ = self.encoder(img_tensor)
                
            if self.attn_input is None:
                raise RuntimeError("ไม่สามารถดึง Attention Input จาก DINOv2 hook ได้")
                
            # คำนวณ self-attention แบบแมนนวลจาก Q และ K เพื่อหลีกเลี่ยงข้อจำกัดของ xFormers/SDPA
            attn_module = self.encoder.blocks[-1].attn
            qkv = attn_module.qkv(self.attn_input) # [1, 257, 384*3]
            B, N, C_three = qkv.shape
            C = C_three // 3
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
            q, k, v = torch.unbind(qkv, 2)
            
            q = q.transpose(1, 2) # [1, num_heads, 257, head_dim]
            k = k.transpose(1, 2) # [1, num_heads, 257, head_dim]
            
            # Compute attention matrix: [1, num_heads, 257, 257]
            scale = head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = scores.softmax(dim=-1)
            
            # cls_attn shape: [1, num_heads, 257, 257] -> ดึงเฉพาะ CLS token ไปหา patch tokens
            cls_attn = attn[:, :, 0, 1:] # [1, num_heads, 256]
            cls_attn = cls_attn.mean(dim=1).squeeze(0) # [256]
                
            heatmap = cls_attn.reshape(16, 16).cpu().numpy()
            
            # Normalize heatmap to [0, 1]
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max - heatmap_min > 1e-8:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap = np.zeros_like(heatmap)
                
            # Resize heatmap to 512x512 for fast contour analysis
            heatmap_resized = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_CUBIC)
            
            # Threshold to get binary mask
            mask = (heatmap_resized > threshold).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_crops = 0
            scale_x = w_org / 512.0
            scale_y = h_org / 512.0
            
            for idx, cnt in enumerate(contours):
                x_512, y_512, w_512, h_512 = cv2.boundingRect(cnt)
                contour_area_512 = cv2.contourArea(cnt)
                total_area_512 = 512 * 512
                
                # Check if this contour area is at least min_area_ratio of the 512x512 area
                if w_512 > 10 and h_512 > 10 and (contour_area_512 / total_area_512 >= min_area_ratio):
                    x_org = max(0, int(x_512 * scale_x))
                    y_org = max(0, int(y_512 * scale_y))
                    w_org_box = min(w_org - x_org, int(w_512 * scale_x))
                    h_org_box = min(h_org - y_org, int(h_512 * scale_y))
                    
                    if w_org_box > 30 and h_org_box > 30:
                        cropped_img = img_org.crop((x_org, y_org, x_org + w_org_box, y_org + h_org_box))
                        
                        # Pad to square to prevent distortion during subsequent resizing
                        max_dim = max(w_org_box, h_org_box)
                        padded_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                        padded_img.paste(cropped_img, ((max_dim - w_org_box) // 2, (max_dim - h_org_box) // 2))
                        
                        filename = os.path.splitext(os.path.basename(img_path))[0]
                        save_name = f"{filename}_leaf_{idx}.jpg"
                        save_class_dir = os.path.join(cleaned_dir, class_name)
                        os.makedirs(save_class_dir, exist_ok=True)
                        padded_img.save(os.path.join(save_class_dir, save_name), quality=95)
                        valid_crops += 1
                        
            if valid_crops > 0:
                return True
            else:
                # No valid leaf found, copy original to discarded directory
                discard_class_dir = os.path.join(discarded_dir, class_name)
                os.makedirs(discard_class_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(discard_class_dir, os.path.basename(img_path)))
                return False
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            try:
                discard_class_dir = os.path.join(discarded_dir, class_name)
                os.makedirs(discard_class_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(discard_class_dir, os.path.basename(img_path)))
            except:
                pass
            return False

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

        # 0. ตรวจหาและตีกรอบใบไม้เพื่อทำความสะอาดข้อมูลก่อน
        cleaned_dir = os.path.join(os.path.dirname(target_dir), "cleaned_leaves")
        discarded_dir = os.path.join(os.path.dirname(target_dir), "discarded_leaves")
        
        # ลบโฟลเดอร์เก่าถ้ามีอยู่
        if os.path.exists(cleaned_dir):
            shutil.rmtree(cleaned_dir)
        if os.path.exists(discarded_dir):
            shutil.rmtree(discarded_dir)
            
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(discarded_dir, exist_ok=True)
        
        print("\n-> ขั้นตอนที่ 0: กำลังตรวจหา ตีกรอบ และคัดกรองใบไม้ด้วย DINOv2 Attention Map (Unsupervised Leaf Cropping)...")
        all_original_images = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    class_name = os.path.basename(root)
                    if "edited" in class_name.lower():
                        continue
                    all_original_images.append((os.path.join(root, file), class_name))
                    
        total_cleaned = 0
        total_discarded = 0
        
        for img_path, class_name in tqdm(all_original_images, desc="วิเคราะห์และตีกรอบใบไม้"):
            success = self.clean_and_crop_image(img_path, class_name, cleaned_dir, discarded_dir)
            if success:
                total_cleaned += 1
            else:
                total_discarded += 1
                
        print(f"✅ ตีกรอบใบไม้สำเร็จ: บันทึกข้อมูลใบไม้ {total_cleaned} รูป | ย้ายไปโฟลเดอร์คัดออก {total_discarded} รูป")
        
        # เปลี่ยน source_dir ให้เป็นโฟลเดอร์รูปภาพที่ทำความสะอาดและ Crop แล้ว
        source_dir = cleaned_dir

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
