import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cudnn.benchmark = True

# 1. เลเยอร์คณิตศาสตร์ ArcFace (เคล็ดลับความแม่นยำสูง)
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        phi = torch.where(cosine > 0, phi, cosine)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# 2. ฟังก์ชันโฟกัสภาพคู่เหมือน (Hard Example Mining)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# 3. ประกอบร่าง Backbone เข้ากับ ArcFace
class ExpertNet(nn.Module):
    def __init__(self, num_classes):
        super(ExpertNet, self).__init__()
        # ใช้ ConvNeXt หรือ ResNet50 เป็นฐานสกัด Features
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # ตัดหัวจำแนกแบบเก่าทิ้ง
        
        # ใส่หัว ArcFace เข้าไปแทนที่
        self.arcface = ArcMarginProduct(in_features, num_classes, s=30.0, m=0.5)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if labels is not None:
            return self.arcface(features, labels) # ตอนเทรนให้ดึงระยะห่าง
        return features # ตอนใช้งานจริงคืนค่าแค่ Features

# 4. ระบบ Training สำหรับ Expert เฉพาะกลุ่ม
class SpecializedExpertTrainer:
    def __init__(self, target_group_path, batch_size=16, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Data Augmentation แบบจัดเต็ม รองรับ High-Resolution
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(448, scale=(0.5, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # โหลดข้อมูลเฉพาะจากกลุ่มที่ระบุ (เช่น เฉพาะ super_group_0)
        self.train_dataset = datasets.ImageFolder(os.path.join(target_group_path, 'train'), transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(target_group_path, 'validation'), transform=self.val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"[{os.path.basename(target_group_path)}] ตรวจพบพืชย่อยในกลุ่มนี้: {self.num_classes} คลาส")

    def train(self, model_save_name, epochs=30):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        model = ExpertNet(num_classes=self.num_classes).to(self.device)
        # เปลี่ยนจาก CrossEntropy เป็น FocalLoss สำหรับ Hard Negative Mining
        criterion = FocalLoss(gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Expert Epoch {epoch+1}/{epochs} [Train]", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # ส่ง Labels เข้าไปให้ ArcFace คำนวณ Margin
                outputs = model(inputs, labels) 
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            scheduler.step()
            epoch_loss = running_loss / len(self.train_dataset)
            history['train_loss'].append(epoch_loss)

            # Validation เช็คความแม่นยำ
            model.eval()
            corrects = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Expert Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # เวลาเทสต์ เราวัดระยะห่าง Cosine เทียบกับ Weight ของ ArcFace โดยตรง
                    features = model(inputs)
                    cosine = F.linear(F.normalize(features), F.normalize(model.arcface.weight))
                    _, preds = torch.max(cosine, 1)
                    corrects += torch.sum(preds == labels.data)

            epoch_acc = float(corrects) / len(self.val_dataset) * 100
            history['val_acc'].append(epoch_acc)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                # เซฟโมเดลของผู้เชี่ยวชาญกลุ่มนี้
                torch.save(model.state_dict(), f"{model_save_name}.pth")
                print(f"   --> เซฟ Weights [ {model_save_name}.pth ] ที่แม่นยำที่สุดแล้ว!")
                
        # สร้างกราฟ Loss & Accuracy
        self._plot_training_curves(history, epochs, model_save_name)
        
        # ประเมินผลและสร้าง Confusion Matrix หลังจากเทรนเสร็จ
        self._evaluate_model(model, val_loader, self.train_dataset.classes, model_save_name)
        
        # สร้าง Grad-CAM Heatmap เป็นตัวอย่างอธิบายโมเดล
        self._generate_gradcam(model, self.val_dataset, model_save_name)

    def _plot_training_curves(self, history, epochs, save_name):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='red', marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy', color='green', marker='o')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_name}_training_curves.png', dpi=300)
        plt.close()
        print(f"✅ บันทึกกราฟการเทรน {save_name}_training_curves.png")

    def _evaluate_model(self, model, dataloader, class_names, save_name):
        print(f"\nกำลังประเมินผล Expert Model ({save_name})...")
        model.load_state_dict(torch.load(f"{save_name}.pth"))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = model(inputs)
                cosine = F.linear(F.normalize(features), F.normalize(model.arcface.weight))
                _, preds = torch.max(cosine, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Classification Report
        report = classification_report(all_labels, all_preds, target_names=class_names)
        with open(f'{save_name}_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ บันทึก {save_name}_classification_report.txt")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Expert Router Confusion Matrix ({save_name})")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(f'{save_name}_confusion_matrix.png', dpi=300)
        plt.close()
        print(f"✅ บันทึกกราฟ {save_name}_confusion_matrix.png")

    def _generate_gradcam(self, model, dataset, save_name):
        print(f"\nกำลังสร้าง Grad-CAM Heatmap ({save_name})...")
        model.load_state_dict(torch.load(f"{save_name}.pth"))
        model.eval()
        
        # Target layer for ResNet50 is usually the last basic block
        target_layers = [model.backbone.layer4[-1]]
        
        # We need a wrapper to compute the output from the backbone since arcface expects labels during train
        class CAMWrapper(nn.Module):
            def __init__(self, expert_net):
                super().__init__()
                self.expert_net = expert_net
            def forward(self, x):
                features = self.expert_net.backbone(x)
                cosine = F.linear(F.normalize(features), F.normalize(self.expert_net.arcface.weight))
                return cosine

        cam = GradCAM(model=CAMWrapper(model), target_layers=target_layers)
        
        # สุ่มรูปภาพ 1 รูปจาก validation set
        if len(dataset) == 0: return
        img_tensor, label = dataset[0]
        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Denormalize image for display
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"Original Image (Class {dataset.classes[label]})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_name}_gradcam_sample.png', dpi=300)
        plt.close()
        print(f"✅ บันทึกรูป Heatmap {save_name}_gradcam_sample.png")

if __name__ == "__main__":
    BASE_DIR = "processed_data/hierarchical_splits"
    
    # ค้นหากลุ่มทั้งหมดที่มีอยู่ (super_group_0, super_group_1, ...)
    if not os.path.exists(BASE_DIR):
        print(f"ไม่พบโฟลเดอร์ {BASE_DIR} กรุณารัน 2split.py ก่อน")
    else:
        all_groups = sorted([d for d in os.listdir(BASE_DIR) if d.startswith('super_group_')])
        
        print(f"พบโฟลเดอร์กลุ่มทั้งหมด {len(all_groups)} กลุ่ม: {all_groups}")
        print("เริ่มทำการเทรน Expert Models ทีละกลุ่มอัตโนมัติ...")
        
        for group_name in all_groups:
            print(f"\n{'='*50}\n🚀 กำลังเริ่มเทรน: {group_name}\n{'='*50}")
            TARGET_GROUP = os.path.join(BASE_DIR, group_name)
            SAVE_NAME = f"expert_{group_name}"
            
            # สร้างและรัน (แนะนำเทรน 30-50 Epochs เพราะคลาสหน้าตาเหมือนกัน ต้องใช้เวลาหาจุดต่าง)
            # เราใช้ 30 Epochs เป็นมาตรฐานตามที่ตกลงกันไว้
            trainer = SpecializedExpertTrainer(target_group_path=TARGET_GROUP, batch_size=16)
            trainer.train(model_save_name=SAVE_NAME, epochs=30)