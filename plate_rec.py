import tensorflow as tf
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont


vocabulary = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
"琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
"B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
"Y", "Z", "-", "[", "]"]
yolo_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
yolo_model.eval()

rec_model = tf.keras.models.load_model('plateRec_model')
# 还原标签（向量->字符串）
def vec2text(vec):
    text = []
    for c in vec:
        if vocabulary[c] != '-':  # 忽略填充字符
            text.append(vocabulary[c])
    return "".join(text)

# 图片路径
image_path = "1.jpg"

# 读取图像
image = cv2.imread(image_path)
results = yolo_model(image)

pred_boxes = results.xyxy[0]
pred_boxes = pred_boxes.tolist()  # 将tensor转换为list

if pred_boxes:
    for box in pred_boxes:
        # 提取车牌区域
        plate_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        
        # 进行去噪处理
        denoised_test_image = cv2.fastNlMeansDenoisingColored(plate_image, None, 10, 10, 7, 21)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        plate_test_image = cv2.dilate(denoised_test_image, kernel)
        
        # resize图片
        test_img = cv2.resize(plate_test_image, (80, 30))
        img_array = tf.expand_dims(test_img, 0)
        predictions = rec_model.predict(img_array)
        txt_labels = vec2text(np.argmax(predictions, axis=2)[0])
        print(txt_labels)

        # 使用PIL渲染中文文本
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype("simhei.ttf", 30)  # simhei.ttf 是黑体字体，您需要确保该字体文件在您的系统中可用
        draw.text((int(box[0]) + 20, int(box[1]) - 40), txt_labels, font=font, fill=(255, 0, 0))

        # 将PIL图像转换回OpenCV格式
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("没有检测到车牌")
