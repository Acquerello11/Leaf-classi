import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None

# โหลดโครงสร้างเลเยอร์ ArcFace ให้ตรงกับตอนเทรน Expert
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

class ExpertNet(nn.Module):
    def __init__(self, num_classes):
        super(ExpertNet, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.arcface = ArcMarginProduct(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        # ช่วง Inference: วัดค่า Cosine Distance ตรงๆ ไม่ต้องคำนวณ Margin
        cosine = F.linear(F.normalize(features), F.normalize(self.arcface.weight))
        return cosine

class DINORouter(nn.Module):
    def __init__(self, num_classes):
        super(DINORouter, self).__init__()
        self.fc = nn.Linear(384, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class PlantInferenceEngine:
    def __init__(self, hierarchical_splits_dir, master_weights_path, expert_weights_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. ตั้งค่าการแปลงไฟล์ภาพ 2 ระดับ (Phase 3 Upgrade)
        self.transform_master = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform_expert = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # โหลด DINOv2 (ใช้สกัดเวกเตอร์ให้นายประตู)
        print("[Engine] กำลังโหลดโมเดล DINOv2...")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()
        
        # 2. ค้นหาโครงสร้างคลาสจากระบบไดเรกทอรี
        self.group_names = sorted([d for d in os.listdir(hierarchical_splits_dir) if d.startswith('super_group_')])
        
        # เก็บรายชื่อคลาสย่อยในแต่ละกลุ่มเพื่อใช้แปลงผลลัพธ์จาก Index เป็นชื่อจริง
        self.group_to_classes_map = {}
        for group_name in self.group_names:
            train_dir = os.path.join(hierarchical_splits_dir, group_name, 'train')
            if os.path.exists(train_dir):
                self.group_to_classes_map[group_name] = sorted(os.listdir(train_dir))

        # 3. โหลด Master Model (นายประตู - DINOv2 Router)
        num_groups = len(self.group_names)
        self.master_model = DINORouter(num_classes=num_groups)
        if os.path.exists(master_weights_path):
            self.master_model.load_state_dict(torch.load(master_weights_path, map_location=self.device))
        else:
            print(f"[Warning] ไม่พบไฟล์น้ำหนัก {master_weights_path} (กรุณารัน 3train.py เพื่อสร้างโมเดลนายประตูก่อน)")
            
        self.master_model.to(self.device)
        self.master_model.eval()

        # 4. โหลด Expert Models ทุกตัวขึ้นสแตนด์บายบน VRAM
        self.experts = {}
        for group_name in self.group_names:
            num_sub_classes = len(self.group_to_classes_map[group_name])
            expert_model = ExpertNet(num_classes=num_sub_classes)
            
            weights_path = os.path.join(expert_weights_dir, f"expert_{group_name}.pth")
            if os.path.exists(weights_path):
                expert_model.load_state_dict(torch.load(weights_path, map_location=self.device))
                expert_model.to(self.device)
                expert_model.eval()
                self.experts[group_name] = expert_model
                print(f"[Engine] โหลดผู้เชี่ยวชาญประจำระบบ: expert_{group_name} สำเร็จ")

    def predict(self, img_path, top_k=3, get_heatmap=False):
        # แปลงรูปภาพต้นทางเป็น Tensor 2 ระดับ
        img_original = Image.open(img_path).convert('RGB')
        tensor_master = self.transform_master(img_original).unsqueeze(0).to(self.device)
        tensor_expert = self.transform_expert(img_original).unsqueeze(0).to(self.device)
        
        predicted_group_name = None
        group_confidence = 0.0
        final_results = [("Unknown", 0.0)]
        heatmap_img = None
        
        with torch.no_grad():
            # ด่านที่ 1: ให้ Master Model ทายกลุ่มใหญ่ก่อน
            features = self.encoder(tensor_master)
            master_logits = self.master_model(features)
            master_probs = F.softmax(master_logits, dim=1).cpu().numpy()[0]
            
            predicted_group_idx = np.argmax(master_probs)
            predicted_group_name = self.group_names[predicted_group_idx]
            group_confidence = master_probs[predicted_group_idx]
            
            # ด่านที่ 2: ส่งต่อให้ Expert Model เฉพาะทาง
            if predicted_group_name in self.experts:
                expert_model = self.experts[predicted_group_name]
                expert_logits = expert_model(tensor_expert)
                expert_probs = F.softmax(expert_logits, dim=1).cpu().numpy()[0]
                
                # ด่านที่ 3: คำนวณความน่าจะเป็นสุทธิ (Ensemble Probability)
                final_results = []
                sub_classes = self.group_to_classes_map[predicted_group_name]
                
                for sub_class_idx, sub_class_name in enumerate(sub_classes):
                    final_score = group_confidence * expert_probs[sub_class_idx] * 100
                    final_results.append((sub_class_name, final_score))
                    
                # จัดอันดับเปอร์เซ็นต์ความคล้ายคลึงสูงสุด
                final_results.sort(key=lambda x: x[1], reverse=True)
                final_results = final_results[:top_k]

        # สร้าง Heatmap ถ้าร้องขอ
        if get_heatmap and GradCAM is not None and predicted_group_name in self.experts:
            expert_model = self.experts[predicted_group_name]
            # Wrapper for GradCAM
            class CAMWrapper(nn.Module):
                def __init__(self, exp_net):
                    super().__init__()
                    self.exp_net = exp_net
                def forward(self, x):
                    feats = self.exp_net.backbone(x)
                    cos = F.linear(F.normalize(feats), F.normalize(self.exp_net.arcface.weight))
                    return cos

            target_layers = [expert_model.backbone.layer4[-1]]
            cam = GradCAM(model=CAMWrapper(expert_model), target_layers=target_layers)
            
            grayscale_cam = cam(input_tensor=tensor_expert, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # แปลงภาพกลับให้อยู่ในโหมด RGB (Undo Normalize)
            img_np = tensor_expert.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            heatmap_img = Image.fromarray(visualization)

        if get_heatmap:
            return predicted_group_name, group_confidence * 100, final_results, heatmap_img
        else:
            return predicted_group_name, group_confidence * 100, final_results

if __name__ == "__main__":
    # เรียกใช้ระบบประมวลผลควบคู่กันทั้งหมด
    engine = PlantInferenceEngine(
        hierarchical_splits_dir="processed_data/hierarchical_splits",
        master_weights_path="master_router_model.pth",
        expert_weights_dir="." # พาธที่เซฟไฟล์ expert_group_X.pth ไว้
    )
    
    # ทดสอบรันภาพจาก Test Set เพื่อดูประสิทธิภาพเชิงลึก
    TEST_IMAGE = "processed_data/data_cleaned/test/class_01/some_test_image.jpg"
    
    if os.path.exists(TEST_IMAGE):
        group, group_conf, species_list = engine.predict(TEST_IMAGE)
        print(f"\n[วิเคราะห์ผลลัพธ์ระบบสายพาน]")
        print(f"คัดกรองเข้ากลุ่มใหญ่: {group} (ความมั่นใจ {group_conf:.2f}%)")
        print(f"สรุปสายพันธุ์ย่อยที่มีเปอร์เซ็นต์ความคล้ายคลึงสูงสุด:")
        for name, percent in species_list:
            print(f" -> สายพันธุ์: {name} | เปอร์เซ็นต์ความแม่นยำ: {percent:.2f}%")