from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import base64
import io
import logging
import clip
from torchvision import transforms
from dwpose import DWposeDetector


app = Flask(__name__)

class OneShotInference:
    def __init__(self, reference_images, model_name='ViT-B/32'):
        self.reference_images = reference_images
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.reference_features = self.generate_reference_features()

    def extract_features(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
        return features.cpu().numpy()

    def generate_reference_features(self):
        reference_features = {}
        for class_name, image_path in self.reference_images.items():
            image = Image.open(image_path).convert('RGB')
            features_list = []
            for angle in [0, 90, 180, 270]:
                rotated_image = transforms.functional.rotate(image, angle)
                features = self.extract_features(rotated_image)
                features_list.append(features)
            mean_features = np.mean(features_list, axis=0)
            reference_features[class_name] = mean_features
        return reference_features

    def calculate_similarity(self, feature, reference_features):
        similarities = {}
        for class_name, ref_feature in reference_features.items():
            similarity = np.dot(feature, ref_feature.T) / (np.linalg.norm(feature) * np.linalg.norm(ref_feature))
            similarities[class_name] = similarity
        return similarities

    def infer(self, test_image):
        test_feature = self.extract_features(test_image)
        similarities = self.calculate_similarity(test_feature, self.reference_features)
        predicted_class = max(similarities, key=similarities.get)
        return predicted_class

# 示例参考图像路径
reference_images12 = {
    '0': './refimg/class12/gesture_0.jpg',
    '1': './refimg/class12/gesture_1.jpg',
    '2': './refimg/class12/gesture_2.jpg',
    '3': './refimg/class12/gesture_3.jpg',
    '4': './refimg/class12/gesture_4.jpg',
    '5': './refimg/class12/gesture_5.jpg',
    '6': './refimg/class12/gesture_6.jpg',
    '7': './refimg/class12/gesture_7.jpg',
    '8': './refimg/class12/gesture_8.jpg',
    '9': './refimg/class12/gesture_9.jpg',
    '10': './refimg/class12/gesture_10.jpg',
    'back': './refimg/class12/gesture_back.jpg',
    # 添加更多类别的参考图像路径
}

# 初始化OneShotInference类
one_shot_inference12 = OneShotInference(reference_images12)

# 示例参考图像路径
reference_images4 = {
    'others': './refimg/class4/gesture_others.jpg',
    'paper': './refimg/class4/gesture_paper.jpg',
    'rock': './refimg/class4/gesture_rock.jpg',
    'scissors': './refimg/class4/gesture_scissors.jpg',
    # 添加更多类别的参考图像路径
}

# 初始化OneShotInference类
one_shot_inference4 = OneShotInference(reference_images4)

# 人手检测模型
pose = DWposeDetector()

# 场景切换
globals()['flag'] = True

def predict_gesture(image):
    global flag
    try:
        # 将PIL图像转换为NumPy数组
        image_np = np.array(image)
        out_img, left_hands, right_hands = pose(image_np)
        H, W, C = image_np.shape

        # 移除等于 [-300, -150] 的点
        left_hands = left_hands[~np.all(left_hands == [-300, -150], axis=-1)]
        right_hands = right_hands[~np.all(right_hands == [-300, -150], axis=-1)]
        if right_hands.shape[0] > 20 and left_hands.shape[0] > 20:
            flag = not flag

        hands = left_hands if left_hands.shape[0] > right_hands.shape[0] else right_hands

        if right_hands.shape[0] > 2:
            x_min = max(0, int(np.min(hands[:, 0]) - 35))
            x_max = min(W, int(np.max(hands[:, 0]) + 35))
            y_min = max(0, int(np.min(hands[:, 1]) - 15))
            y_max = min(H, int(np.max(hands[:, 1]) + 15))
            
            # 对图像进行切片操作
            hand_image_np = image_np[y_min:y_max, x_min:x_max, :]

            # 将NumPy数组转换回PIL图像
            hand_image = Image.fromarray(hand_image_np)
            with torch.no_grad():
                if flag:
                    gesture = one_shot_inference12.infer(hand_image)
                else:
                    gesture = one_shot_inference4.infer(hand_image)
                
        else:
            x_min = 30
            x_max = W - 30
            y_min = 30
            y_max = H - 30
            
            gesture = "others" if flag else "back"

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