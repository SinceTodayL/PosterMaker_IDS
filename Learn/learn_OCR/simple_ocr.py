import easyocr
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 读取图像（你可以替换为自己的图片路径）
image_path = Path(__file__).parent / 'tokenizer_process.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 创建 OCR 识别器（以英文和中文为例）
reader = easyocr.Reader(['en', 'ch_sim'])  # 你可以替换为其它语言代码

# 3. 执行 OCR：返回检测框 + 文本 + 置信度
results = reader.readtext(image)

# 4. 打印识别结果并可视化检测框
for (bbox, text, confidence) in results:
    print(f"Detected text: {text} (Confidence: {confidence:.2f})")
    
    # 可视化：在图像上画出识别框
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# 5. 显示结果图像
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('OCR Result')
plt.show()
