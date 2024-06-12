import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchsummary import summary
import time


class dir_BirdsDataset(Dataset):
    def __init__(self, path_dir, image_size, transform=None):
        self.path_dir = path_dir
        self.image_size = image_size
        self.transform = transform

        # 从 path_dir 读取所有图像文件路径和相应标签    Прочтите все пути к файлам изображений и соответствующие теги из path_dir
        self.image_paths = []
        self.labels = []
        for class_name in os.listdir(path_dir):
            class_path = os.path.join(path_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_name)

        # 创建类名与数字标签的映射关系    Создайте связь между именами классов и числовыми метками
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

        # 检查过滤后的数据（数量）    Проверить отфильтрованные данные
        print(f"Found {len(self.image_paths)} images in directory '{self.path_dir}'.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        # 输出图片路径调试信息    Выведите путь к изображению
        # print(f"Attempting to load image: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



def dir_load_dataset(path_dir, batch_size, image_size, shuffle):
    transform = transforms.Compose([
        transforms.Resize(image_size),    # Resize images to the same size
        transforms.ToTensor(),            # Convert to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = dir_BirdsDataset(path_dir=path_dir,
                           image_size=image_size,
                           transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    class_to_name = {idx: class_name for class_name, idx in dataset.class_to_idx.items()}

    return data_loader, class_to_name



class CSV_BirdsDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size, split, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        self.transform = transform

        # 根据split过滤数据    Фильтрация данных по split
        self.data_frame = self.data_frame[self.data_frame['data set'] == self.split]

        # 检查过滤后的数据框    Проверить отфильтрованные данные
        print(f"Filtered dataframe based on split '{self.split}':")
        print(self.data_frame.head())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 构建图像文件的完整路径    Полный путь к файлу образа
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1].strip().replace("/", os.sep))  # 统一路径分隔符

        # 输出图片路径调试信息    Выведите путь к изображению
        print(f"Attempting to load image: {img_name}")

        # 处理文件文件名的拼写错误
        if not os.path.exists(img_name):
            corrected_img_name = img_name.replace("AKULET", "AUKLET")
            #print(f"Corrected image name to: {corrected_img_name}")
            if os.path.exists(corrected_img_name):
                img_name = corrected_img_name
            else:
                raise FileNotFoundError(f"File not found: {img_name}")

        image = Image.open(img_name).convert('RGB')     # 打开路径对应的图片
        label = self.data_frame.iloc[idx, 0]        # 储存图像对应的标签，即位于第几行

        # 如果需要对图像进行转换操作，如缩放、裁剪、翻转等
        if self.transform:
            image = self.transform(image)

        return image, label


def CSV_load_dataset(path, batch_size, image_size, shuffle, split):
    transform = transforms.Compose([
        transforms.Resize(image_size),    # 归一化图像尺寸
        transforms.ToTensor(),        # 统一转化成PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CSV_BirdsDataset(csv_file=path,
                           root_dir=os.path.dirname(path),
                           image_size=image_size,
                           split=split,
                           transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 创建类别映射字典    Создайте связь между именами классов и числовыми метками
    class_to_name = {row["class id"]: row["labels"] for _, row in pd.read_csv(path).iterrows()}

    return data_loader, class_to_name


path = "C:\\Users\\lcf14\\Desktop\\archive\\birds.csv"
path_dir = "C:\\Users\\lcf14\\Desktop\\archive\\train"
batch_size = 64
image_size = (224, 224)
shuffle = True
split = "train"

# 只能选择其中一种方法来运行!!!
#data_loader, class_to_name = CSV_load_dataset(path, batch_size, image_size, shuffle, split)
data_loader, class_to_name = dir_load_dataset(path_dir, batch_size, image_size, shuffle)

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_to_name))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def training(data_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch running time: {epoch_time} seconds")

    accuracy = 100 * correct / total
    print(f"Training loss: {(running_loss / len(data_loader)):.3f}%, Train accuracy: {accuracy:.3f}%")

    return running_loss, accuracy


train_loss, train_accuracy = training(data_loader, model, criterion, optimizer)
