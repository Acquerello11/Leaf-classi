import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from Modeltrainer import RobustPlantPredictor

def evaluate_validation_set():
    # กำหนด Path
    model_path = 'dino_model.pkl'
    val_dir = 'processed_data/data_cleaned/validation'
    
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
        return
    if not os.path.exists(val_dir):
        print(f"❌ ไม่พบโฟลเดอร์ Validation Data: {val_dir}")
        return

    print("🧠 กำลังโหลดโมเดล DINOv2 และ SVM...")
    predictor = RobustPlantPredictor(model_path)
    if predictor.classifier is None:
        print("❌ เกิดข้อผิดพลาดในการโหลด Classifier")
        return

    # คลาสทั้งหมดเรียงตามตัวอักษร
    classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    
    y_true = []
    y_pred = []
    
    print(f"📂 กำลังทดสอบโมเดลกับข้อมูล Validation ({len(classes)} คลาส)...")
    
    for cls in classes:
        cls_path = os.path.join(val_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            
            # ทำนายผล (Top 1)
            predictions = predictor.predict(img_path, top_k=1)
            
            if predictions:
                predicted_class = predictions[0][0]
                y_true.append(cls)
                y_pred.append(predicted_class)
            else:
                print(f"⚠️ ไม่สามารถทำนายรูปภาพได้: {img_path}")

    # คำนวณความแม่นยำ
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ ทดสอบเสร็จสิ้น! จำนวนรูปภาพทั้งหมดที่ประเมิน: {len(y_true)} รูป")
    print(f"🎯 ความแม่นยำรวม (Overall Accuracy): {acc * 100:.2f}%\n")
    
    # แสดง Classification Report
    print("📊 Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # สร้าง Confusion Matrix
    print("🎨 กำลังสร้างกราฟ Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    fig, ax = plt.subplots(figsize=(24, 24))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Greens', ax=ax, xticks_rotation='vertical', values_format='d')
    
    plt.title(f'Validation Confusion Matrix (Accuracy: {acc * 100:.2f}%)', fontsize=20, pad=20)
    plt.ylabel('Actual Label (ความจริง)', fontsize=16)
    plt.xlabel('Predicted Label (โมเดลทำนาย)', fontsize=16)
    plt.tight_layout()
    
    # บันทึกภาพ Confusion Matrix
    output_image = 'validation_confusion_matrix.png'
    plt.savefig(output_image, dpi=200)
    print(f"💾 บันทึกรูป Confusion Matrix เรียบร้อยแล้วที่: {output_image}")

if __name__ == "__main__":
    evaluate_validation_set()
