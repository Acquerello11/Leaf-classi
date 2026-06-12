import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# เปิดโหมดรีดความเร็วสูงสุดของการ์ดจอ
cudnn.benchmark = True

# 1. สร้าง Custom Dataset เพื่อดึง Label เป็นชื่อกลุ่มใหญ่ (Super Group)
class SuperGroupDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.filepaths = []
        self.labels = []
        self.transform = transform
        
        # ค้นหาว่ามีกี่กลุ่มใหญ่ (เช่น super_group_0 ถึง 4)
        self.group_names = sorted([d for d in os.listdir(root_dir) if d.startswith('super_group_')])
        self.num_classes = len(self.group_names)

        # สแกนเข้าไปใน super_group_X -> split -> class_Y
        for group_idx, group_name in enumerate(self.group_names):
            split_dir = os.path.join(root_dir, group_name, split)
            if not os.path.exists(split_dir): continue

            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir): continue

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.filepaths.append(os.path.join(class_dir, img_name))
                        self.labels.append(group_idx) # ใช้ ID ของกลุ่มใหญ่เป็น Label

    def __len__(self): 
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class DINORouter(nn.Module):
    def __init__(self, num_classes):
        super(DINORouter, self).__init__()
        # DINOv2-vits14 ให้เวกเตอร์ขนาด 384 มิติ
        self.fc = nn.Linear(384, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class MasterModelTrainer:
    def __init__(self, data_root, batch_size=32, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root
        self.batch_size = batch_size
        self.lr = lr
        
        print(f"กำลังโหลดโมเดล DINOv2 (Feature Extractor) บน {self.device}...")
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval() # แช่แข็ง DINOv2 ไม่ต้องอัปเดตน้ำหนัก

        # กำหนด Data Augmentation แบบเบาๆ
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train(self, epochs=15):
        print("กำลังเตรียมข้อมูลสำหรับ Master Model...")
        train_dataset = SuperGroupDataset(self.data_root, split='train', transform=self.train_transform)
        val_dataset = SuperGroupDataset(self.data_root, split='validation', transform=self.val_transform)
        
        num_groups = train_dataset.num_classes
        print(f"ตรวจพบกลุ่มใหญ่ทั้งหมด: {num_groups} กลุ่ม")
        
        # โหมด Max Speed (ยอมเครื่องค้างตอนเริ่ม): ใช้คนงาน 8 คน, pin_memory, persistent, โหลดล่วงหน้า 4 เท่า
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        # ใช้ Linear Layer ตัวเล็กๆ เป็นนายประตูแทน EfficientNet
        model = DINORouter(num_classes=num_groups).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_acc = 0.0
        
        # สำหรับเก็บประวัติการเทรน
        history = {'train_loss': [], 'val_acc': []}
        
        print(f"\nเริ่มเทรน Master Model (Linear Probe) บนอุปกรณ์: {self.device}")
        for epoch in range(epochs):
            # --- Training ---
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # สกัดเวกเตอร์ด้วย DINOv2 (ไม่ต้องคิด Gradient ทำให้ไวมาก)
                with torch.no_grad():
                    features = self.encoder(inputs)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            history['train_loss'].append(epoch_loss)

            # --- Validation ---
            model.eval()
            corrects = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    features = self.encoder(inputs)
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    corrects += torch.sum(preds == labels.data)
                    
            epoch_acc = float(corrects) / len(val_dataset) * 100
            history['val_acc'].append(epoch_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'master_router_model.pth')
                print("   --> บันทึกโมเดลนายประตูที่ดีที่สุดแล้ว!")
                
        # สร้างกราฟ Loss & Accuracy
        self._plot_training_curves(history, epochs)
        
        # ประเมินผลและสร้าง Confusion Matrix หลังจากเทรนเสร็จ
        self._evaluate_model(model, val_loader, train_dataset.group_names)

    def _plot_training_curves(self, history, epochs):
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
        plt.savefig('master_training_curves.png', dpi=300)
        plt.close()
        print("✅ บันทึกกราฟการเทรน master_training_curves.png เรียบร้อยแล้ว")

    def _evaluate_model(self, model, dataloader, class_names):
        print("\nกำลังประเมินผล Master Model...")
        model.load_state_dict(torch.load('master_router_model.pth'))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = self.encoder(inputs)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Classification Report
        report = classification_report(all_labels, all_preds, target_names=class_names)
        with open('master_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✅ บันทึก master_classification_report.txt เรียบร้อยแล้ว")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title("Master Router Confusion Matrix")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig('master_confusion_matrix.png', dpi=300)
        plt.close()
        print("✅ บันทึกกราฟ master_confusion_matrix.png เรียบร้อยแล้ว")

if __name__ == "__main__":
    # ใส่พาธที่คุณเก็บโฟลเดอร์ hierarchical_splits เอาไว้
    DATA_ROOT = "processed_data/hierarchical_splits"
    
    # ขยาย batch_size เป็น 128 เพื่อใช้ประโยชน์จากการ์ดจอ 8GB ให้คุ้มค่าที่สุด
    trainer = MasterModelTrainer(data_root=DATA_ROOT, batch_size=128)
    trainer.train(epochs=15)