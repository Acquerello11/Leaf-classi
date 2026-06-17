# ใช้ Base Image ของ PyTorch 2.4.0 คู่กับ CUDA 12.1 (เพื่อให้รองรับ Python 3.12 และ GPU)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ตั้งค่าตำแหน่งโฟลเดอร์หลักภายใน Container
WORKDIR /app

# อัปเดตและติดตั้งไลบรารีระบบพื้นฐานที่จำเป็น (สำหรับ OpenCV และอื่นๆ)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ requirements.txt เข้าไปก่อน (เทคนิคนี้ช่วยให้ Docker จำ Cache ประหยัดเวลาเวลา Build ใหม่)
COPY requirements.txt .

# รันคำสั่งติดตั้งไลบรารี Python
RUN pip install --no-cache-dir -r requirements.txt

# หมายเหตุ: เราจะไม่คัดลอกไฟล์โค้ดหรือรูปภาพ (.py, processed_data) เข้าไปในอิมเมจนี้
# แต่เราจะใช้วิธี "Volume Mount" เชื่อมโฟลเดอร์จากเครื่องจริงเข้าไปแทน เพื่อให้เซฟโมเดลลงเครื่องจริงได้ทันที
