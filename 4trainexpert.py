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
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                # เซฟโมเดลของผู้เชี่ยวชาญกลุ่มนี้
                torch.save(model.state_dict(), f"{model_save_name}.pth")
                print(f"   --> เซฟ Weights [ {model_save_name}.pth ] ที่แม่นยำที่สุดแล้ว!")

if __name__ == "__main__":
    # --- วิธีการรัน: คุณต้องเปลี่ยน TARGET_GROUP ไปเรื่อยๆ จนครบทุกกลุ่ม ---
    
    BASE_DIR = "processed_data/hierarchical_splits"
    
    # รอบที่ 1: เทรน Expert สำหรับกลุ่มที่ 0
    TARGET_GROUP = os.path.join(BASE_DIR, "super_group_0")
    SAVE_NAME = "expert_group_0"
    
    # สร้างและรัน (แนะนำเทรน 30-50 Epochs เพราะคลาสหน้าตาเหมือนกัน ต้องใช้เวลาหาจุดต่าง)
    trainer = SpecializedExpertTrainer(target_group_path=TARGET_GROUP, batch_size=16)
    trainer.train(model_save_name=SAVE_NAME, epochs=30)