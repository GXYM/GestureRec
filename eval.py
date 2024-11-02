import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 数据预处理
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/apdcephfs_cq10/share_1367250/somoszhang/GestureRec/datas010'
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
dataloaders = DataLoader(image_datasets, batch_size=32, shuffle=False, num_workers=4)
class_names = image_datasets.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet-101模型
model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

# 加载训练好的模型权重
model_ft.load_state_dict(torch.load('gesture_resnet101.pth'))

# 评估模型
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))

    for i in range(len(class_names)):
        if class_total[i] > 0:
            print('Accuracy of {} : {:.4f}'.format(class_names[i], class_correct[i] / class_total[i]))
        else:
            print('Accuracy of {} : N/A (no samples)'.format(class_names[i]))

    return total_loss, total_acc

criterion = nn.CrossEntropyLoss()
evaluate_model(model_ft, dataloaders, criterion)