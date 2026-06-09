import gradio as gr
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

# Configuration
MODEL_PATH = 'best_leaf_model_b1_finetuned.keras'
TRAIN_DIR = 'processed_data/data/train'
IMG_SIZE = (224, 224)

# 1. Load Class Names
if os.path.exists(TRAIN_DIR):
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
else:
    class_names = [f"Class {i}" for i in range(61)]

# 2. Load Model
model = None
if os.path.exists(MODEL_PATH):
    print("🧠 กำลังโหลดสมอง AI...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ โหลดสมอง AI สำเร็จ!")
    except Exception as e:
        print(f"⚠️ ไฟล์ '{MODEL_PATH}' ยังสร้างไม่เสร็จสมบูรณ์")
        model = None
else:
    print(f"⚠️ ยังไม่พบไฟล์ '{MODEL_PATH}'")

# 3. Single prediction
def predict_single(pil_img):
    img = pil_img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    top_5_indices = predictions.argsort()[-5:][::-1]
    top_class = class_names[top_5_indices[0]]
    top5 = {class_names[i]: float(predictions[i]) for i in top_5_indices}
    return top_class, top5

# 4. Multi-image prediction → cards HTML + store results
def predict_multiple(files):
    if model is None:
        return ("<div style='text-align:center;padding:40px;color:#c62828;'>❌ โมเดลยังไม่พร้อม</div>",
                [], gr.update(visible=False))
    if not files:
        return ("<div style='text-align:center;padding:40px;color:#888;'>กรุณาอัปโหลดรูปภาพ</div>",
                [], gr.update(visible=False))

    all_results = []
    cards_html = ""
    for file_data in files:
        try:
            pil_img = Image.open(file_data).convert("RGB")
            top_class, top5 = predict_single(pil_img)
            fname = os.path.basename(file_data) if isinstance(file_data, str) else "image"
            all_results.append({"file": fname, "predicted": top_class})

            # Convert thumbnail to base64 for inline display
            thumb = pil_img.copy()
            thumb.thumbnail((200, 200))
            buf = io.BytesIO()
            thumb.save(buf, format='JPEG', quality=80)
            buf.seek(0)
            thumb_b64 = base64.b64encode(buf.read()).decode('utf-8')

            bars = ""
            first = True
            for name, prob in top5.items():
                pct = prob * 100
                bar_color = "#2e7d32" if first else "#81c784"
                wt = "bold" if first else "normal"
                bars += f"""<div style="margin-bottom:5px;">
                  <div style="display:flex;justify-content:space-between;font-size:0.85em;font-weight:{wt};color:#333;">
                    <span>{name}</span><span>{pct:.1f}%</span></div>
                  <div style="background:#e8f5e9;border-radius:6px;overflow:hidden;height:10px;">
                    <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:6px;"></div>
                  </div></div>"""
                first = False

            top_prob = list(top5.values())[0] * 100
            cards_html += f"""
            <div style="background:white;border:1px solid #c8e6c9;border-radius:16px;padding:16px;
                        box-shadow:0 2px 8px rgba(46,125,50,0.06);display:flex;gap:16px;margin-bottom:12px;">
              <div style="flex-shrink:0;width:140px;height:140px;border-radius:10px;overflow:hidden;border:2px solid #e8f5e9;">
                <img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;height:100%;object-fit:cover;" /></div>
              <div style="flex:1;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:0.75em;color:#888;">{fname}</div>
                <div style="font-size:1.2em;font-weight:bold;color:#2e7d32;">🌿 {top_class} ({top_prob:.1f}%)</div>
                <div style="margin-top:8px;">{bars}</div>
              </div></div>"""
        except Exception as e:
            cards_html += f"<div style='background:#fff3e0;border:1px solid #ffcc80;border-radius:12px;padding:16px;margin-bottom:12px;'>⚠️ {e}</div>"

    header = f"""<div style="background:#e8f5e9;border-radius:12px;padding:14px;margin-bottom:14px;text-align:center;">
      <span style="font-size:1.1em;color:#2e7d32;">📊 ทำนายเสร็จ <strong>{len(all_results)}</strong> รูป</span></div>"""

    return header + cards_html, all_results, gr.update(visible=True)

# 5. Add batch to cumulative confusion data & generate matrix
def add_to_confusion(actual_class, current_results, cumulative_data):
    if not current_results or not actual_class:
        return cumulative_data, "<div style='color:#c62828;padding:20px;text-align:center;'>❌ กรุณาทำนายรูปและเลือกคลาสจริงก่อน</div>"

    # Add current batch to cumulative data
    if cumulative_data is None:
        cumulative_data = []

    for r in current_results:
        cumulative_data.append({"actual": actual_class, "predicted": r["predicted"]})

    # Build confusion matrix from all cumulative data
    involved_classes = sorted(set([d["actual"] for d in cumulative_data] + [d["predicted"] for d in cumulative_data]))
    n = len(involved_classes)
    cls_to_idx = {c: i for i, c in enumerate(involved_classes)}

    cm = np.zeros((n, n), dtype=int)
    for d in cumulative_data:
        cm[cls_to_idx[d["actual"]]][cls_to_idx[d["predicted"]]] += 1

    total = len(cumulative_data)
    correct = sum(1 for d in cumulative_data if d["actual"] == d["predicted"])
    overall_acc = correct / total * 100

    # Count per actual class
    class_counts = {}
    for d in cumulative_data:
        cls = d["actual"]
        if cls not in class_counts:
            class_counts[cls] = {"total": 0, "correct": 0}
        class_counts[cls]["total"] += 1
        if d["predicted"] == cls:
            class_counts[cls]["correct"] += 1

    # Plot confusion matrix
    fig_size = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Normalize by row
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm_norm / row_sums * 100

    im = ax.imshow(cm_pct, cmap='Greens', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fontsize = max(6, min(12, 120 // n))
    ax.set_xticklabels(involved_classes, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(involved_classes, fontsize=fontsize)
    ax.set_xlabel('Predicted (AI ทำนาย)', fontsize=11)
    ax.set_ylabel('Actual (ความจริง)', fontsize=11)
    ax.set_title(f'Confusion Matrix — Accuracy: {overall_acc:.1f}%  ({correct}/{total})', fontsize=13, fontweight='bold', pad=15)

    # Add text in cells
    cell_fontsize = max(5, min(11, 80 // n))
    for i in range(n):
        for j in range(n):
            count = cm[i][j]
            pct = cm_pct[i][j]
            if count > 0:
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{count}\n({pct:.0f}%)', ha='center', va='center',
                        color=color, fontsize=cell_fontsize, fontweight='bold' if i == j else 'normal')

    plt.colorbar(im, ax=ax, label='%', shrink=0.8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')

    # Per-class accuracy summary table
    rows = ""
    for cls in involved_classes:
        if cls in class_counts:
            c = class_counts[cls]
            acc = c["correct"] / c["total"] * 100
            bar_color = "#2e7d32" if acc >= 80 else "#ef6c00" if acc >= 50 else "#c62828"
            rows += f"""<tr>
              <td style="padding:6px 12px;border-bottom:1px solid #eee;">{cls}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;">{c['total']}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;">{c['correct']}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:right;color:{bar_color};font-weight:bold;">{acc:.1f}%</td>
            </tr>"""

    html = f"""
    <div style="text-align:center;margin:20px 0;">
      <div style="font-size:2em;font-weight:bold;color:#2e7d32;">📊 Overall Accuracy: {overall_acc:.1f}%</div>
      <div style="color:#666;margin-top:4px;">({correct}/{total} รูป, {len(involved_classes)} คลาส)</div>
      <div style="color:#888;font-size:0.85em;margin-top:4px;">💡 เพิ่มรูปคลาสอื่นได้เรื่อยๆ ตารางจะสะสมผลให้อัตโนมัติ</div>
    </div>

    <div style="text-align:center;margin:20px 0;">
      <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.1);" />
    </div>

    <div style="max-width:500px;margin:20px auto;">
      <h4 style="color:#2e7d32;text-align:center;">📋 สรุปแต่ละคลาส</h4>
      <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
        <thead><tr style="background:#e8f5e9;">
          <th style="padding:8px 12px;text-align:left;">คลาส</th>
          <th style="padding:8px 12px;text-align:center;">จำนวน</th>
          <th style="padding:8px 12px;text-align:center;">ถูก</th>
          <th style="padding:8px 12px;text-align:right;">แม่นยำ</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return cumulative_data, html

# 6. Reset cumulative data
def reset_confusion():
    return None, "<div style='text-align:center;padding:30px;color:#aaa;'>เริ่มต้นใหม่แล้ว — อัปโหลดรูปเพื่อเริ่มวัดผล</div>"

# 7. Theme
custom_theme = gr.themes.Soft(
    primary_hue="green", secondary_hue="emerald", neutral_hue="stone",
).set(
    body_background_fill="white", body_background_fill_dark="#1a261d",
    block_background_fill="#f2fbf4", block_label_background_fill="*primary_100",
    block_label_text_color="*primary_700", button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

# 8. Build UI
with gr.Blocks(title="🌿 Leaf Classification AI") as app:
    prediction_state = gr.State([])
    cumulative_state = gr.State(None)

    gr.Markdown("""<div style="text-align:center;padding:16px 0 8px;">
      <h1 style="color:#2e7d32;font-size:2.3em;margin:0;">🌿 ระบบ AI จำแนกสายพันธุ์ใบไม้</h1>
      <p style="color:#666;margin-top:6px;">อัปโหลดรูป → ทำนาย → เลือกคลาสจริง → สร้าง Confusion Matrix</p>
    </div>""")

    # Step 1: Upload & Predict
    gr.Markdown("### ขั้นตอนที่ 1: อัปโหลดรูปและทำนาย")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", file_types=["image"],
                                 label="📥 อัปโหลดรูปภาพ (ลากวางได้หลายรูป)")
            predict_btn = gr.Button("🔍 ทำนายผลทั้งหมด", variant="primary", size="lg")

    result_html = gr.HTML(value="<div style='text-align:center;padding:30px;color:#aaa;'>อัปโหลดรูปแล้วกดปุ่ม 🔍</div>")

    # Step 2: Select actual class & add to confusion matrix
    with gr.Group(visible=False) as accuracy_section:
        gr.Markdown("### ขั้นตอนที่ 2: เลือกคลาสจริงและวัดผล")
        gr.Markdown("<p style='color:#666;font-size:0.9em;'>เลือกว่ารูปชุดนี้เป็นใบไม้คลาสไหนจริงๆ แล้วกดเพิ่มเข้า Confusion Matrix</p>")
        with gr.Row():
            with gr.Column(scale=2):
                class_dropdown = gr.Dropdown(choices=class_names, label="🏷️ คลาสจริง (Actual Class)")
            with gr.Column(scale=1):
                add_btn = gr.Button("➕ เพิ่มเข้า Confusion Matrix", variant="primary", size="lg")

    # Step 3: Confusion Matrix Display
    gr.Markdown("### 📊 Confusion Matrix (สะสมผลจากทุกชุดที่เพิ่ม)")
    cm_html = gr.HTML(value="<div style='text-align:center;padding:30px;color:#aaa;'>ยังไม่มีข้อมูล — อัปโหลดรูปแล้วเพิ่มเข้า Confusion Matrix</div>")
    reset_btn = gr.Button("🗑️ เริ่มใหม่ (Reset)", variant="secondary")

    # Events
    predict_btn.click(fn=predict_multiple, inputs=file_input,
                      outputs=[result_html, prediction_state, accuracy_section])
    add_btn.click(fn=add_to_confusion, inputs=[class_dropdown, prediction_state, cumulative_state],
                  outputs=[cumulative_state, cm_html])
    reset_btn.click(fn=reset_confusion, inputs=[], outputs=[cumulative_state, cm_html])

if __name__ == "__main__":
    print("🌐 กำลังเปิดเซิร์ฟเวอร์หน้าเว็บ...")
    app.launch(inbrowser=True, theme=custom_theme, share=True)
