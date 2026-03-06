import streamlit as st
import torch
import numpy as np
import os
import cv2
import io
from PIL import Image
import torch.nn.functional as Fnn
from google.genai import types
from transforms import JointSegTransform
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# Import các hàm từ file của bạn
from finetune import get_model 
from prompt import client # Đảm bảo file prompt.py đã cấu hình client genai

# --- CẤU HÌNH ---
MY_PALETTE = {
    (0,0,0): 0,
    (240,240,13): 1,
    (233,117,5): 2,
    (8,182,8): 3,    
}
ID2COLOR = {v: k for k, v in MY_PALETTE.items()}
CLASS_NAMES = ['Màng nhĩ phải bình thường', 'Màng nhĩ trái bình thường', 'Viêm tai giữa cấp', 'Viêm tai giữa mạn tính', 'Viêm tai giữa tiết dịch' ]


def get_transform(image):
    image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
    image = TF.to_tensor(image)
    image = TF.convert_image_dtype(image, torch.float32)
    return image


# --- HÀM HỖ TRỢ ---
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(4, 5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, device

def draw_polygon_overlay(img_np, mask_np, alpha=0.4):
    if torch.is_tensor(mask_np):
        mask_np = mask_np.cpu().numpy()
    
    overlay_fill = img_np.copy()
    img_out = img_np.copy()

    for class_id in range(1, 4): # 1, 2, 3
        class_mask = (mask_np == class_id).astype(np.uint8) * 255
        if np.sum(class_mask) == 0: continue

        kernel = np.ones((3,3), np.uint8)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = ID2COLOR.get(class_id, (255, 255, 255))
        cv2.fillPoly(overlay_fill, contours, color)
        cv2.polylines(img_out, contours, isClosed=True, color=color, thickness=2)

    result = cv2.addWeighted(overlay_fill, alpha, img_out, 1 - alpha, 0)
    return result

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="AI Endoscopy Diagnosis", layout="wide")
st.title("🩺 Hệ thống phân tích nội soi tai")
st.sidebar.header("Cấu hình")

checkpoint_path = st.sidebar.text_input("Đường dẫn Checkpoint", "/mnt/mmlab2024nas/khanhnq/check_point_deeplabv3/exp_c2/best_model.pth")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh nội soi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load Model
    with st.spinner('Đang tải mô hình...'):
        
        model, device = load_model(checkpoint_path)

    # 2. Đọc ảnh
    img_pil = Image.open(uploaded_file).convert("RGB")
    org_w, org_h = img_pil.size
    img_np = np.array(img_pil)

    # 3. Tiền xử lý & Dự đoán
    # transform = get_transform()
    input_tensor = get_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        seg_output, pred_cls = outputs['out'], outputs['cls']
        
        # Lấy nhãn phân loại
        cls_idx = torch.softmax(pred_cls, dim=1).argmax(dim=1).item()
        
        # Upscale mask
        logits_up = Fnn.interpolate(seg_output, size=(org_h, org_w), mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(logits_up, dim=1).squeeze(0).cpu()

    # 4. Tạo ảnh Overlay
    overlay_res = draw_polygon_overlay(img_np, pred_mask)
    Image.fromarray(overlay_res).save(os.path.join('/home/khanhnq/Experiment/Mask_RCNN/res_image',  f"overlay_1.png"))
    # 5. Hiển thị ảnh
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader("Kết quả Segmentation")
        st.image(overlay_res, use_container_width=True)

    # THÊM CHÚ THÍCH MÀU (LEGEND) Ở ĐÂY
    st.markdown("### 💡 Chú thích các vùng màu:")
    cols = st.columns(3)

    # Định nghĩa CSS cho các "nút" chú thích
    def color_block(color_hex, text):
        return f"""
        <div style="
            display: flex; 
            align-items: center; 
            background-color: #f0f2f6; 
            padding: 10px; 
            border-radius: 10px; 
            border-left: 10px solid {color_hex};
            margin-bottom: 10px;">
            <span style="font-weight: bold; color: #31333F; margin-left: 10px;">{text}</span>
        </div>
        """

    with cols[0]:
        st.markdown(color_block("#00FF08", "Cán xương búa (Xanh)"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(color_block("#FFA500", "Tam giác sáng (Cam)"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(color_block("#FFFF00", "Bóng khí (Vàng)"), unsafe_allow_html=True)

    # 6. Gọi Gemini API
    st.divider()
    st.subheader("🤖 Phân tích ")
    
    with st.spinner('Đang phân tích...'):
        # Chuyển ảnh overlay sang bytes
        img_byte_arr = io.BytesIO()
        Image.fromarray(overlay_res).save(img_byte_arr, format='PNG')
        image_part = types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/png")

        prompt =  """
            Bạn là một chuyên gia nội soi Tai Mũi Họng giàu kinh nghiệm.
            
            NHIỆM VỤ: Phân tích ảnh màng nhĩ dựa trên mask màu:
            Mask xanh lá: Cán búa
            Mask cam: Tam giác sáng
            Mask vàng: Bóng khí
            
            QUY TẮC PHẢN HỒI:
            Chỉ trả lời TRÊN 1 DÒNG DUY NHẤT.
            Nếu có mask nào trên hình, phải ghi chú màu mask ngay sau tên bộ phận đó trong mô tả, nếu không có thì không ghi chú vào mô tả.
            Ví dụ:
            Có: "tam giác sáng rõ (mask cam), cán xương búa thấy rõ (mask xanh) ".
            Không có: "Tam giác sáng mất, cán xương búa mất".
            
            Chỉ chọn 1 trong các mẫu dưới đây để điền thông tin:
            Mẫu 1: Mô Tả: Màng nhĩ nguyên vẹn, tam giác sáng rõ , cán xương búa thấy rõ
            Mẫu 2: Mô Tả: Màng nhĩ nguyên vẹn, tam giác sáng không thấy, cán xương búa thấy rõ
            Mẫu 3: Mô Tả: Màng nhĩ căng phồng/sung huyết, tam giác sáng mất/còn , mất/còn vị trí cán xương búa
            Mẫu 4: Mô Tả: Màng nhĩ màu hổ phách/ bóng khí sau màng nhĩ , tam giác mất/còn , cán xương búa còn/ mất .
            Mẫu 5: Mô Tả: Màng nhĩ thủng, tam giác sáng còn/ mất , cán xương búa còn/ mất .
            
            Lưu ý: "/" là chọn 1 trong 2. Không để lại dấu "/" trong kết quả cuối cùng.
            """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro", # Cập nhật model name nếu cần
                contents=[prompt, image_part],
            )
            
            # HIỂN THỊ KẾT QUẢ CUỐI CÙNG
            st.info(f"**Kết quả chẩn đoán (Model):** {CLASS_NAMES[cls_idx]} (ID: {cls_idx})")
            st.success(f" {response.text}")
            
        except Exception as e:
            st.error(f"Lỗi khi gọi Gemini API: {e}")

else:
    st.info("Vui lòng tải ảnh lên từ thanh bên để bắt đầu.")