import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image

class OneShotInference:
    def __init__(self, num_classes, reference_images, model_path):
        self.num_classes = num_classes
        self.reference_images = reference_images
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self.load_model()
        self.reference_features = self.generate_reference_features()

    def load_model(self):
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        model = model.to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        return model

    def extract_features(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            features = self.model(image)
            features = features.view(features.size(0), -1)
        return features.cpu().numpy()

    def generate_reference_features(self):
        reference_features = {}
        for class_name, image_path in self.reference_images.items():
            image = Image.open(image_path).convert('RGB')
            features_list = []
            for angle in [0, 90, 180, 270]:
                rotated_image = transforms.functional.rotate(image, angle)
                rotated_image = self.data_transforms(rotated_image).unsqueeze(0)
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

    def infer(self, test_image_path):
        test_image = Image.open(test_image_path).convert('RGB')
        test_image = self.data_transforms(test_image).unsqueeze(0)
        test_feature = self.extract_features(test_image)
        similarities = self.calculate_similarity(test_feature, self.reference_features)
        predicted_class = max(similarities, key=similarities.get)
        return predicted_class

# 示例参考图像路径
reference_images = {
    'class1': '/path/to/reference_image1.jpg',
    'class2': '/path/to/reference_image2.jpg',
    # 添加更多类别的参考图像路径
}

# 初始化OneShotInference类
num_classes = len(reference_images)
model_path = 'gesture_resnet101.pth'
one_shot_inference = OneShotInference(num_classes, reference_images, model_path)

# 推理示例
test_image_path = '/path/to/test_image.jpg'
predicted_class = one_shot_inference.infer(test_image_path)
print(f'The predicted class for the test image is: {predicted_class}')