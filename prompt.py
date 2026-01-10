import os
import cv2
from google import genai
from google.genai import types
import io # Import io for BytesIO
from pathlib import Path
import mimetypes
from PIL import Image

# Directly provide the API key string
API_KEY = "AIzaSyBIMwe0ysz9ViSsqdkxCh7pyJiT3PO54vQ"
client = genai.Client(api_key=API_KEY)

# 2. Tạo đối tượng Path từ đường dẫn
image_path = "/home/khanhnq/Experiment/Mask_RCNN/res_image/overlay_OME_52.png.png"
path = Path(image_path)

if not path.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

with Image.open(image_path) as image:
    # Chuyển đổi sang RGB nếu cần (để đảm bảo tính tương thích cao nhất)
    image = image.convert('RGB')

# 4. Lưu ảnh vào một bộ nhớ đệm (BytesIO) dưới dạng PNG
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG', compress_level=0)
image_bytes = img_byte_arr.getvalue()
# 5. Tạo đối tượng Part cho API
image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

# 6. Gửi yêu cầu
response = client.models.generate_content(
    model="gemini-3-pro-image-preview", # Hoặc model bạn đang sử dụng
    contents=[
        '''
        Bạn là một chuyên gia nội soi Tai Mũi Họng giàu kinh nghiệm.
        Dựa vào hình ảnh nội soi tai tôi cung cấp, hãy đưa ra mô tả trên 1 dòng, với mask xanh là cán búa, mask cam là tam giác sáng, mask vàng là dịch-bóng khí.
        Hãy chú thích thêm màu tương ứng của từng phần trong mô tả, nếu không có, không cần chú thích.
        Dựa vào các cách trả lời sau :
        Mô tả: Màng nhĩ nguyên vẹn, tam giác sáng rõ, cán xương búa thấy rõ
        Mô Tả: Màng nhĩ căng phồng, tam giác sáng mất , mất vị trí cán búa
        Mô Tả: Màng nhĩ căng phồng, tam giác sáng còn , mất vị trí cán búa
        Mô Tả: Màng nhĩ màu hổ phách/ bóng khí sau màng nhĩ, tam giác mất, cán xương búa còn/ mất.
        Mô Tả: Màng nhĩ thủng trung tâm/ rộng, còn rìa/ sát rìa, cán xương búa còn/ mất, tam giác sáng còn/ mất.

        Lưu ý: "/" là thể hiện lựa chọn hoặc cái này hoặc cái kia
        ''',
        image_part,
    ],
)

print(response.text)