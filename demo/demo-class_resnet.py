from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import base64
import io
import logging
from dwpose import DWposeDetector


app = Flask(__name__)

# 加载模型
model10 = models.resnet101()
num_ftrs = model10.fc.in_features
model10.fc = torch.nn.Linear(num_ftrs, 12)  # 假设有5个手势类别
model10.load_state_dict(torch.load('./gesture_resnet101_010_90.pth', map_location=torch.device('cuda')))
model10.eval()


# 加载模型
model4 = models.resnet101()
num_ftrs = model4.fc.in_features
model4.fc = torch.nn.Linear(num_ftrs, 4)  # 假设有5个手势类别
model4.load_state_dict(torch.load('./gesture_resnet101_0c3_4.pth', map_location=torch.device('cuda')))
model4.eval()

pose = DWposeDetector()

# 数据预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 类别名称
class_names10 = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'back']

# 类别名称
class_names4 = ['others', 'paper', 'rock', 'scissors']

# 场景切换
globals()['flag'] = True

def predict_gesture(image):
    global flag
    try:
        # 将PIL图像转换为NumPy数组
        image_np = np.array(image)
        out_img, left_hands,  right_hands = pose(image_np)
        H, W, C = image_np.shape

        # 移除等于 [-300, -150] 的点
        left_hands = left_hands[~np.all(left_hands == [-300, -150], axis=-1)]
        right_hands = right_hands[~np.all(right_hands == [-300, -150], axis=-1)]
        if right_hands.shape[0] > 20 and left_hands.shape[0] > 20:
            flag = (not flag)

        hands = left_hands if left_hands.shape[0] > right_hands.shape[0] else right_hands

        if right_hands.shape[0] > 2:
            x_min = max(0, int(np.min(hands[:, 0]) - 25))
            x_max = min(W, int(np.max(hands[:, 0]) + 25))
            y_min = max(0, int(np.min(hands[:, 1]) - 15))
            y_max = min(H, int(np.max(hands[:, 1]) + 15))
            
            # 对图像进行切片操作
            hand_image_np = image_np[y_min:y_max, x_min:x_max, :]

            # 将NumPy数组转换回PIL图像
            hand_image = Image.fromarray(hand_image_np)

            img_t = preprocess(hand_image)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                if flag:
                    out = model10(batch_t)
                    _, predicted = torch.max(out, 1)
                    gesture = class_names10[predicted[0]]
                else:
                    out = model4(batch_t)
                    _, predicted = torch.max(out, 1)
                    gesture = class_names4[predicted[0]]
                
        else:
            x_min =30
            x_max = W -30
            y_min = 30
            y_max = H -30
            
            gesture = class_names4[0] if flag else class_names10[-1]

        print(y_min, y_max, x_min, x_max)
        # 将图像从RGB转换为BGR格式
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # 在原始图像上绘制边界框和手势识别结果
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image_np, gesture, (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 将结果图像转换为Base64编码的字符串
        _, buffer = cv2.imencode('.jpg', image_np)
        result_image_str = base64.b64encode(buffer).decode('utf-8')

        return gesture, result_image_str
    except Exception as e:
        logging.error(f"Error in predict_gesture: {e}")
        return "error", None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        gesture, result_image_str = predict_gesture(image)
        if result_image_str is None:
            return jsonify({'gesture': 'error'})
        return jsonify({'gesture': gesture, 'image': result_image_str})
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({'gesture': 'error'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))
