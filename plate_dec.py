import os
import cv2
import torch

plate_path = 'dataset_DY/plate_images/train_process/'

img_path = 'dataset_DY/Images/train/'  

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model.eval()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
for img_file in os.listdir(img_path):
# for img_file in range(1):
    img = cv2.imread(os.path.join(img_path, img_file))
    # img = cv2.imread("dataset_DY/Images/train/2.jpg")
    try:
        # 使用yolov5模型预测
        plates = model(img)   

        pred_boxes = plates.xyxy[0]  # 获取第一个预测结果的边界框坐标
        pred_boxes = pred_boxes.tolist()  # 将tensor转换为list
        # print(pred_boxes)
        if pred_boxes :
            # print("yes")
            for box in pred_boxes:
                plate_image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                denoised_image = cv2.fastNlMeansDenoisingColored(plate_image, None, 10, 10, 7, 21) # 进行去噪处理
                plate_images = cv2.dilate(denoised_image, kernel)
                # plate_image = cv2.resize(plate_image, (80, 30))
                # print(plate_image.shape)
            # 保存文件名不变
                # cv2.imshow("test",plate_image)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(plate_path, img_file), plate_images)
                # cv2.imwrite("dataset_DY/plate_images/train/2.jpg", plate_image)
                print(img_file)
    except AttributeError as e:
        pass
