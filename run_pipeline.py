import subprocess
import sys

# เรียงลำดับไฟล์ที่ต้องรันต่อกัน
scripts = [
    "1_2_gradio_cleaner.py",
    "3train.py",
    "4trainexpert.py",
    "6evaluate.py"
]

print("🌟 กำลังเริ่มระบบ Auto-Pipeline รันทุกไฟล์รวดเดียวจบ 🌟")

for script in scripts:
    print(f"\n{'='*50}")
    print(f"🚀 กำลังรันไฟล์: {script}")
    print(f"{'='*50}\n")
    
    # รันคำสั่ง Python ผ่าน Terminal จำลอง
    result = subprocess.run([sys.executable, script])
    
    # ถ้าไฟล์ไหนรันแล้ว Error (Code ไม่ใช่ 0) ให้หยุดทำงานทันที
    if result.returncode != 0:
        print(f"\n❌ หยุดการทำงาน! พบข้อผิดพลาดขณะรันไฟล์ {script}")
        sys.exit(1)

print("\n🎉🎊 ไพป์ไลน์ทั้งหมดทำงานเสร็จสมบูรณ์ 100% แล้ว! 🎊🎉")
