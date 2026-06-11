import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report
import importlib.util
from tqdm import tqdm

# โหลดโมดูลจากไฟล์ที่ชื่อขึ้นต้นด้วยตัวเลข (5mixmodel.py)
spec = importlib.util.spec_from_file_location("mixmodel", "5mixmodel.py")
mixmodel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixmodel)
PlantInferenceEngine = mixmodel.PlantInferenceEngine

def evaluate_system(engine, test_dir):
    print("เริ่มทำการทดสอบระบบประมวลผลสายพันธุ์พืช (Evaluation)...")
    
    true_labels = []
    predicted_labels = []
    all_class_names = []
    
    # สแกนหาชื่อคลาสทั้งหมดที่มีในโฟลเดอร์ Test
    for group_name in os.listdir(test_dir):
        group_path = os.path.join(test_dir, group_name, 'test')
        if not os.path.isdir(group_path): continue
            
        for class_name in os.listdir(group_path):
            if class_name not in all_class_names:
                all_class_names.append(class_name)
    
    all_class_names.sort()

    # รันทำนายภาพทุกภาพในโฟลเดอร์ Test
    total_images = 0
    correct_predictions = 0

    # เตรียมรูปทั้งหมดก่อนส่งเข้า tqdm
    all_test_images = []
    for group_name in os.listdir(test_dir):
        group_path = os.path.join(test_dir, group_name, 'test')
        if not os.path.isdir(group_path): continue

        for class_name in os.listdir(group_path):
            class_dir = os.path.join(group_path, class_name)
            if not os.path.isdir(class_dir): continue

            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
                img_path = os.path.join(class_dir, img_name)
                all_test_images.append((class_name, img_path))

    for class_name, img_path in tqdm(all_test_images, desc="กำลังรันโมเดลทำนายผล (Evaluation)"):
        true_labels.append(class_name)
        total_images += 1

        # เรียกใช้งานระบบ Inference 
        try:
            predicted_group, group_conf, top_results = engine.predict(img_path, top_k=1)
            predicted_class = top_results[0][0] # เอาอันดับ 1 ที่ AI มั่นใจที่สุด
            predicted_labels.append(predicted_class)
            
            if predicted_class == class_name:
                correct_predictions += 1
        except Exception as e:
            print(f"เกิดข้อผิดพลาดกับภาพ {img_path}: {e}")
                    predicted_labels.append("error")

    # สรุปผลความแม่นยำรวม (Overall Accuracy)
    overall_accuracy = (correct_predictions / total_images) * 100
    print(f"\nประมวลผลเสร็จสิ้น {total_images} ภาพ")
    print(f"*** ความแม่นยำสุทธิของระบบ (Overall Accuracy): {overall_accuracy:.2f}% ***")

    # สร้างรายงานสถิติแยกตามคลาส (Precision, Recall, F1-Score)
    print("\n--- รายงานเชิงลึก (Classification Report) ---")
    print(classification_report(true_labels, predicted_labels, target_names=all_class_names))

    # สร้างกราฟ Confusion Matrix
    print("\nกำลังสร้างกราฟ Confusion Matrix...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_class_names)
    
    plt.figure(figsize=(20, 16)) # ปรับขนาดภาพให้รองรับ 61 คลาส
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
    plt.title('Confusion Matrix - Plant Identification (61 Classes)')
    plt.xlabel('Predicted Species')
    plt.ylabel('True Species')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix_result.png')
    print("บันทึกกราฟลงไฟล์ 'confusion_matrix_result.png' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    # ระบุพิกัดที่ตั้งของไฟล์และโมเดลทั้งหมด
    ENGINE = PlantInferenceEngine(
        hierarchical_splits_dir="processed_data/hierarchical_splits",
        master_weights_path="master_router_model.pth",
        expert_weights_dir="." # พาธที่เก็บไฟล์ expert_group_X.pth
    )
    
    TEST_DATA_DIR = "processed_data/hierarchical_splits"
    
    # สั่งประเมินผล
    evaluate_system(ENGINE, TEST_DATA_DIR)