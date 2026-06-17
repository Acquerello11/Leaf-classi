import os
import shutil
import random
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# บังคับไม่ใช้ xFormers เพื่อให้เรียกใช้การคำนวณ Attention map แบบดั้งเดิมได้สะดวกขึ้น
os.environ["XFORMERS_DISABLED"] = "1"

class ImageListDataset(Dataset):
    """ Custom dataset for loading a list of images """
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, index):
        path = self.filepaths[index]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, path, True
        except Exception as e:
            # คืนค่า dummy tensor หากไฟล์เสียหาย
            return torch.zeros(3, 224, 224), path, False

class GradioLeafCleaner:
    def __init__(self, source_dir='Data200', target_dir='processed_data/hierarchical_splits'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.cleaned_dir = os.path.join(os.path.dirname(target_dir), "cleaned_leaves")
        self.discarded_dir = os.path.join(os.path.dirname(target_dir), "discarded_leaves")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"กำลังโหลดระบบวิเคราะห์โครงสร้างเวกเตอร์ (DINOv2) บน: {self.device}")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()
        
        # ลงทะเบียน Hook เพื่อคำนวณ Attention Map
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
        
        # สแกนหาไฟล์ทั้งหมด
        self.classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and "edited" not in d.lower()])
        self.image_list = [] # ลิสต์ไฟล์ทั้งหมดในรูปทรง: (img_path, class_name)
        for cls in self.classes:
            cls_dir = os.path.join(source_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_list.append((os.path.join(cls_dir, f), cls))
                    
        print(f"ตรวจพบทั้งหมด: {len(self.classes)} คลาส | รวมรูปภาพดิบ: {len(self.image_list)} รูป")
        
        # ตัวแปรควบคุมการทำงานของแอปพลิเคชัน
        self.current_idx = 0
        self.decisions = {} # เก็บผลลัพธ์ของแต่ละภาพ: img_path -> 'keep' / 'discard'
        self.current_class_data = {} # เก็บคะแนน similarity ของภาพในคลาสปัจจุบัน
        self.cached_class_name = None
        
    def get_class_embeddings_and_median(self, class_name):
        """ สกัดเวกเตอร์ทุกรูปในคลาสปัจจุบันเพื่อคำนวณหาค่ามัธยฐานและ Cosine Similarity """
        if self.cached_class_name == class_name:
            return
            
        print(f"-> กำลังสกัดเวกเตอร์เปรียบเทียบในคลาส: {class_name}...")
        cls_dir = os.path.join(self.source_dir, class_name)
        filepaths = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        dataset = ImageListDataset(filepaths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        
        embs = []
        paths = []
        
        with torch.no_grad():
            for imgs, pths, valids in dataloader:
                # เอาเฉพาะไฟล์ที่โหลดสำเร็จ
                mask = valids
                if not any(mask): continue
                imgs = imgs[mask].to(self.device)
                
                features = self.encoder(imgs).flatten(1).cpu().numpy()
                embs.append(features)
                for idx, valid in enumerate(mask):
                    if valid:
                        paths.append(pths[idx])
                        
        if len(embs) > 0:
            embs = np.concatenate(embs)
            median_emb = np.median(embs, axis=0)
            similarities = cosine_similarity(embs, median_emb.reshape(1, -1)).flatten()
            
            self.current_class_data = dict(zip(paths, similarities))
        else:
            self.current_class_data = {}
            
        self.cached_class_name = class_name
        
    def analyze_image(self, img_path, class_name, threshold=0.5, min_area_ratio=0.02):
        """ ตรวจสอบพิกัดใบไม้และวิเคราะห์สถิติต่างๆ """
        self.get_class_embeddings_and_median(class_name)
        
        # 1. อ่านรูปภาพและตรวจสอบความสมบูรณ์
        try:
            img_org = Image.open(img_path).convert('RGB')
        except Exception as e:
            return None, "คัดออก", 0.0, "ไฟล์ภาพชำรุด", []
            
        w_org, h_org = img_org.size
        img_np = np.array(img_org)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 2. รัน DINOv2 เพื่อดึง attention map
        img_tensor = self.scan_transform(img_org).unsqueeze(0).to(self.device)
        self.attn_input = None
        with torch.no_grad():
            _ = self.encoder(img_tensor)
            
        if self.attn_input is None:
            return img_np, "เก็บไว้", 50.0, "ไม่สามารถดึง attention map", []
            
        with torch.no_grad():
            attn_module = self.encoder.blocks[-1].attn
            qkv = attn_module.qkv(self.attn_input)
            B, N, C_three = qkv.shape
            C = C_three // 3
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
            q, k, v = torch.unbind(qkv, 2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            scale = head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = scores.softmax(dim=-1)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1).squeeze(0) # [256]
            
        heatmap = cls_attn.reshape(16, 16).detach().cpu().numpy()
        
        # Normalize heatmap
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
            
        heatmap_resized = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask = (heatmap_resized > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        scale_x = w_org / 512.0
        scale_y = h_org / 512.0
        
        valid_leaves = []
        for idx, cnt in enumerate(contours):
            x_512, y_512, w_512, h_512 = cv2.boundingRect(cnt)
            contour_area_512 = cv2.contourArea(cnt)
            total_area_512 = 512 * 512
            
            if w_512 > 10 and h_512 > 10 and (contour_area_512 / total_area_512 >= min_area_ratio):
                x_org = max(0, int(x_512 * scale_x))
                y_org = max(0, int(y_512 * scale_y))
                w_org_box = min(w_org - x_org, int(w_512 * scale_x))
                h_org_box = min(h_org - y_org, int(h_512 * scale_y))
                
                if w_org_box > 30 and h_org_box > 30:
                    valid_leaves.append((x_org, y_org, w_org_box, h_org_box))
                    
        # 3. คำนวณคะแนนคุณภาพ
        sim_score = self.current_class_data.get(img_path, 0.5)
        # แปลง cosine similarity (ประมาณ 0.5 - 1.0) ให้อยู่ในช่วง 0 - 100%
        class_consistency = max(0.0, min(100.0, (sim_score - 0.5) * 200.0))
        
        saliency_score = 100.0 if len(valid_leaves) > 0 else 20.0
        quality_score = 0.6 * class_consistency + 0.4 * saliency_score
        
        # ตัดเกณฑ์การแนะนำ
        is_good = quality_score >= 60.0
        recommendation = "เก็บไว้" if is_good else "คัดออก"
        
        # 4. วาดการแสดงผล
        color = (0, 255, 0) if is_good else (0, 0, 255) # เขียว = เก็บ, แดง = คัดออก
        
        # สร้าง Heatmap Overlay สีตามคำแนะนำ
        # เขียว-แดงผสมใน heatmap
        heatmap_color = np.zeros((h_org, w_org, 3), dtype=np.uint8)
        heatmap_resized_full = cv2.resize(heatmap, (w_org, h_org), interpolation=cv2.INTER_CUBIC)
        
        if is_good:
            # แชนแนลสีเขียวเด่น
            heatmap_color[..., 1] = (heatmap_resized_full * 255).astype(np.uint8)
        else:
            # แชนแนลสีแดงเด่น
            heatmap_color[..., 2] = (heatmap_resized_full * 255).astype(np.uint8)
            
        img_visualized = cv2.addWeighted(img_cv, 0.7, heatmap_color, 0.4, 0)
        
        for idx, (x, y, w, h) in enumerate(valid_leaves):
            cv2.rectangle(img_visualized, (x, y), (x + w, y + h), color, 3)
            label_text = f"Leaf {idx}: Good" if is_good else f"Leaf {idx}: Low Quality"
            cv2.putText(img_visualized, label_text, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        reason = ""
        if len(valid_leaves) == 0:
            reason += "ตรวจไม่พบวัตถุเด่นชัด (Saliency ต่ำ) | "
        else:
            reason += f"ตรวจพบใบไม้ทั้งหมด {len(valid_leaves)} ใบ | "
            
        if class_consistency >= 70.0:
            reason += "ภาพมีความสอดคล้องกับพืชคลาสนี้อย่างมาก"
        else:
            reason += "ภาพมีลักษณะแปลกแยก (Outlier) กว่ารูปอื่นๆ ในคลาสนี้"
            
        img_out = cv2.cvtColor(img_visualized, cv2.COLOR_BGR2RGB)
        return img_out, recommendation, quality_score, reason, valid_leaves

    def apply_decision_and_save(self, img_path, class_name, decision, valid_leaves):
        """ บันทึกการตัดสินใจของผู้ใช้และสกัด/ย้ายรูปจริง """
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        if decision == 'keep':
            # บันทึกภาพลงโฟลเดอร์ภาพดี
            save_class_dir = os.path.join(self.cleaned_dir, class_name)
            os.makedirs(save_class_dir, exist_ok=True)
            
            try:
                img_org = Image.open(img_path).convert('RGB')
                
                # หากสกัดได้ใบไม้ดี
                if len(valid_leaves) > 0:
                    for idx, (x, y, w, h) in enumerate(valid_leaves):
                        cropped_img = img_org.crop((x, y, x + w, y + h))
                        # ทำ Padding สี่เหลี่ยมจัตุรัส
                        max_dim = max(w, h)
                        padded_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                        padded_img.paste(cropped_img, ((max_dim - w) // 2, (max_dim - h) // 2))
                        padded_img.save(os.path.join(save_class_dir, f"{filename}_leaf_{idx}.jpg"), quality=95)
                else:
                    # หากวิเคราะห์พิกัดไม่สำเร็จแต่ยืนยันจะเก็บ ให้ย้ายรูปต้นฉบับไปเลย
                    shutil.copy2(img_path, os.path.join(save_class_dir, os.path.basename(img_path)))
            except Exception as e:
                print(f"Error saving image: {e}")
        else:
            # ย้ายไปโฟลเดอร์ลบ/คัดออก
            discard_class_dir = os.path.join(self.discarded_dir, class_name)
            os.makedirs(discard_class_dir, exist_ok=True)
            try:
                shutil.copy2(img_path, os.path.join(discard_class_dir, os.path.basename(img_path)))
            except Exception as e:
                print(f"Error moving to discarded: {e}")

    def run_auto_process_remaining(self):
        """ รันกระบวนการตัดสินใจด้วย AI อัตโนมัติทั้งหมดสำหรับไฟล์ที่เหลืออยู่ """
        remaining_count = len(self.image_list) - self.current_idx
        print(f"-> กำลังทำงานอัตโนมัติบนภาพที่เหลือทั้งหมด {remaining_count} รูป...")
        
        batch_size = 64
        
        # ทำการประมวลผลเป็นกลุ่ม
        for i in range(self.current_idx, len(self.image_list), batch_size):
            batch_items = self.image_list[i : i + batch_size]
            
            # 1. สกัด Similarity ในคลาสสำหรับแบทช์
            for img_path, class_name in batch_items:
                self.get_class_embeddings_and_median(class_name)
                
            # 2. โหลดรูปและ stack
            batch_tensors = []
            valid_items = []
            
            for img_path, class_name in batch_items:
                try:
                    img_org = Image.open(img_path).convert('RGB')
                    valid_items.append((img_path, class_name, img_org))
                    img_tensor = self.scan_transform(img_org)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    # ข้ามไฟล์ที่เสีย
                    try:
                        discard_class_dir = os.path.join(self.discarded_dir, class_name)
                        os.makedirs(discard_class_dir, exist_ok=True)
                        shutil.copy2(img_path, os.path.join(discard_class_dir, os.path.basename(img_path)))
                    except:
                        pass
                        
            if not batch_tensors:
                continue
                
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            B_size = batch_tensor.shape[0]
            
            self.attn_input = None
            with torch.no_grad():
                _ = self.encoder(batch_tensor)
                if self.attn_input is None:
                    continue
                    
                attn_module = self.encoder.blocks[-1].attn
                qkv = attn_module.qkv(self.attn_input)
                B, N, C_three = qkv.shape
                C = C_three // 3
                num_heads = attn_module.num_heads
                head_dim = C // num_heads
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                q, k, v = torch.unbind(qkv, 2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                scale = head_dim ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale
                attn = scores.softmax(dim=-1)
                cls_attn = attn[:, :, 0, 1:].mean(dim=1)
                
            heatmaps = cls_attn.reshape(B_size, 16, 16).detach().cpu().numpy()
            
            for idx, (img_path, class_name, img_org) in enumerate(valid_items):
                w_org, h_org = img_org.size
                heatmap = heatmaps[idx]
                
                heatmap_min = heatmap.min()
                heatmap_max = heatmap.max()
                if heatmap_max - heatmap_min > 1e-8:
                    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
                else:
                    heatmap = np.zeros_like(heatmap)
                    
                heatmap_resized = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_CUBIC)
                mask = (heatmap_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_leaves = []
                scale_x = w_org / 512.0
                scale_y = h_org / 512.0
                
                for cnt in contours:
                    x_512, y_512, w_512, h_512 = cv2.boundingRect(cnt)
                    contour_area_512 = cv2.contourArea(cnt)
                    total_area_512 = 512 * 512
                    
                    if w_512 > 10 and h_512 > 10 and (contour_area_512 / total_area_512 >= 0.02):
                        x_org = max(0, int(x_512 * scale_x))
                        y_org = max(0, int(y_512 * scale_y))
                        w_org_box = min(w_org - x_org, int(w_512 * scale_x))
                        h_org_box = min(h_org - y_org, int(h_512 * scale_y))
                        
                        if w_org_box > 30 and h_org_box > 30:
                            valid_leaves.append((x_org, y_org, w_org_box, h_org_box))
                            
                # คะแนนความใกล้ชิด
                sim_score = self.current_class_data.get(img_path, 0.5)
                class_consistency = max(0.0, min(100.0, (sim_score - 0.5) * 200.0))
                saliency_score = 100.0 if len(valid_leaves) > 0 else 20.0
                quality_score = 0.6 * class_consistency + 0.4 * saliency_score
                
                # ทำการตัดสินใจอัตโนมัติ
                decision = 'keep' if quality_score >= 60.0 else 'discard'
                self.decisions[img_path] = decision
                self.apply_decision_and_save(img_path, class_name, decision, valid_leaves)
                
        self.current_idx = len(self.image_list)
        return "สำเร็จ! ประมวลผลภาพที่เหลือเสร็จเรียบร้อยแล้ว!"

    def run_split_pipeline(self):
        """ รันกระบวนการแบ่งข้อมูล Train/Val/Test ต่อเมื่อผู้ใช้กดเสร็จสิ้น """
        print("\n-> กำลังเตรียมแบ่งข้อมูลฝึกฝนพืชพรรณและจัดกลุ่ม Super Class...")
        
        # สุ่มและล้างโฟลเดอร์ splits เดิม
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir, exist_ok=True)
        
        # รันสคริปต์สแกนโมเดล DINOv2 อีกครั้งเพื่อสกัดคุณสมบัติและจับกลุ่ม Super groups จากโฟลเดอร์ cleaned_leaves
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # ซึ่งเราจะดึงตัวประมวลผลเดิมของ split มาทำงาน
        import importlib
        prepare_and_split = importlib.import_module("1_2_prepare_and_split")
        OriginalSplitProcessor = prepare_and_split.UnifiedDataProcessor
        
        processor = OriginalSplitProcessor()
        # ข้ามกระบวนการ clean_and_crop ของเดิมเนื่องจากเราทำความสะอาดผ่าน UI เสร็จหมดแล้ว
        # โดยการตั้งค่า source_dir ชี้ตรงไปยัง cleaned_leaves
        
        # จำลองการเรียกฟังก์ชัน
        processor.process(
            source_dir=self.cleaned_dir,
            target_dir=self.target_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            min_groups=12,
            max_groups=25
        )
        print("🎉 แบ่งชุดข้อมูลและจัดกลุ่มพืช super_groups สำเร็จเรียบร้อย 100%!")

# เริ่มต้นระบบ Gradio
cleaner = GradioLeafCleaner()

def get_current_state():
    """ แสดงสถานะและวิเคราะห์รูปปัจจุบัน """
    if cleaner.current_idx >= len(cleaner.image_list):
        return None, "วิเคราะห์เสร็จสิ้นทั้งหมดแล้ว!", "สำเร็จ", "0.0%", "กรุณากดปุ่มสับหลีก/แบ่งชุดข้อมูล (Finish & Split) เพื่อเทรนต่อ", None
        
    img_path, class_name = cleaner.image_list[cleaner.current_idx]
    filename = os.path.basename(img_path)
    
    img_out, rec, q_score, reason, valid_leaves = cleaner.analyze_image(img_path, class_name)
    
    status_text = f"รูปภาพที่ {cleaner.current_idx + 1} จาก {len(cleaner.image_list)} | คลาส: {class_name}"
    score_text = f"{q_score:.2f}%"
    
    rec_html = f"<b style='color:green;font-size:20px;'>{rec}</b>" if rec == "เก็บไว้" else f"<b style='color:red;font-size:20px;'>{rec}</b>"
    
    return img_out, status_text, rec_html, score_text, reason, valid_leaves

def handle_keep(valid_leaves):
    if cleaner.current_idx < len(cleaner.image_list):
        img_path, class_name = cleaner.image_list[cleaner.current_idx]
        cleaner.decisions[img_path] = 'keep'
        cleaner.apply_decision_and_save(img_path, class_name, 'keep', valid_leaves)
        cleaner.current_idx += 1
    return get_current_state()

def handle_discard(valid_leaves):
    if cleaner.current_idx < len(cleaner.image_list):
        img_path, class_name = cleaner.image_list[cleaner.current_idx]
        cleaner.decisions[img_path] = 'discard'
        cleaner.apply_decision_and_save(img_path, class_name, 'discard', valid_leaves)
        cleaner.current_idx += 1
    return get_current_state()

def handle_auto():
    msg = cleaner.run_auto_process_remaining()
    state = get_current_state()
    return state[0], state[1], state[2], state[3], f"{msg} | {state[4]}", state[5]

def handle_finish():
    cleaner.run_split_pipeline()
    return "✅ กระบวนการล้างข้อมูลและแบ่งชุดข้อมูล (Train/Val/Test) เสร็จสิ้นสมบูรณ์แล้ว! คุณสามารถปิดหน้านี้และไปเริ่มเทรนโมเดลต่อได้เลยครับ"

# สร้างหน้าตาเว็บแอปด้วย Gradio
with gr.Blocks(title="Leaf AI Data Cleaner & Bounding Box Visualizer") as demo:
    gr.Markdown("# 🍃 Leaf AI Data Cleaner & Bounding Box Visualizer")
    gr.Markdown("ระบบวิเคราะห์ขอบเขตใบไม้ด้วย DINOv2 และประเมินความสอดคล้องของรูปภาพพืช เพื่อคัดกรองข้อมูลเสียออกและ Crop เพิ่มขนาดชุดข้อมูลที่สมบูรณ์")
    
    # ตัวแปรซ่อนของ Gradio สำหรับส่งค่าลิสต์ใบไม้
    leaves_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=2):
            img_view = gr.Image(label="ภาพตรวจวิเคราะห์ขอบเขตใบไม้ (สีเขียว=แนะนำให้เก็บ, สีแดง=แนะนำให้ลบ)", interactive=False)
            status_info = gr.Textbox(label="สถานะรูปภาพปัจจุบัน", value="กำลังเริ่มระบบ...", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 รายละเอียดการประเมินจาก AI")
            rec_display = gr.HTML(label="ข้อเสนอแนะจากระบบ (Recommendation)")
            score_display = gr.Textbox(label="คะแนนความสมบูรณ์ของพืช (Quality Score)", interactive=False)
            reason_info = gr.Textbox(label="เหตุผลวิเคราะห์ (AI Reasoning)", interactive=False)
            
            with gr.Row():
                btn_keep = gr.Button("✅ OK เก็บไว้ (Keep)", variant="primary")
                btn_discard = gr.Button("❌ ย้ายไปโฟลเดอร์ลบ (Discard)", variant="stop")
                
            gr.Markdown("---")
            btn_auto = gr.Button("🤖 ประมวลผลรูปที่เหลือทั้งหมดอัตโนมัติ", variant="secondary")
            btn_finish = gr.Button("🚀 บันทึกและแบ่งชุดข้อมูล (Finish & Split)", variant="primary")
            
            output_finish = gr.Textbox(label="รายงานผลลัพธ์ขั้นตอนสุดท้าย", interactive=False)

    # เชื่อมโยง Event ต่างๆ
    demo.load(get_current_state, outputs=[img_view, status_info, rec_display, score_display, reason_info, leaves_state])
    
    btn_keep.click(handle_keep, inputs=[leaves_state], outputs=[img_view, status_info, rec_display, score_display, reason_info, leaves_state])
    btn_discard.click(handle_discard, inputs=[leaves_state], outputs=[img_view, status_info, rec_display, score_display, reason_info, leaves_state])
    btn_auto.click(handle_auto, outputs=[img_view, status_info, rec_display, score_display, reason_info, leaves_state])
    btn_finish.click(handle_finish, outputs=[output_finish])

if __name__ == "__main__":
    # เปิดเซิร์ฟเวอร์แบบสาธารณะชั่วคราวผ่านแชร์ลิงก์ของ Gradio (เนื่องจากรันบน Docker และไม่ได้แมปพอร์ต)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
