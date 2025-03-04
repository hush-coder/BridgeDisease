import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定义裂缝检测CNN模型
class CrackDetectionCNN(nn.Module):
    def __init__(self):
        super(CrackDetectionCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)  # 二分类：有裂缝/无裂缝
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 输入图像预期为 3x256x256
        x = self.pool(F.relu(self.conv1(x)))  # 输出: 32x128x128
        x = self.pool(F.relu(self.conv2(x)))  # 输出: 64x64x64
        x = self.pool(F.relu(self.conv3(x)))  # 输出: 128x32x32
        
        # 展平
        x = x.view(-1, 128 * 32 * 32)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 自定义数据集类
class BridgeCrackDataset(Dataset):
    def __init__(self, images_path, labels, transform=None):
        self.images_path = images_path
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据预处理和加载函数
def load_data(data_dir, batch_size=32):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 假设数据目录结构为：
    # data_dir/
    #    - crack/      (含有裂缝的图像)
    #    - no_crack/   (不含裂缝的图像)
    
    crack_dir = os.path.join(data_dir, 'crack')
    no_crack_dir = os.path.join(data_dir, 'no_crack')
    
    # 收集图像路径和标签
    image_paths = []
    labels = []
    
    # 有裂缝的图像
    for img_name in os.listdir(crack_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(crack_dir, img_name))
            labels.append(1)  # 1 表示有裂缝
    
    # 无裂缝的图像
    for img_name in os.listdir(no_crack_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(no_crack_dir, img_name))
            labels.append(0)  # 0 表示无裂缝
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据集
    train_dataset = BridgeCrackDataset(train_paths, train_labels, transform)
    val_dataset = BridgeCrackDataset(val_paths, val_labels, transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

# 训练模型函数
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

# 预测/推理函数
def predict(model, image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        
    result = "裂缝" if predicted.item() == 1 else "无裂缝"
    
    # 显示图像和预测结果
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(image_path))
    plt.title(f'预测结果: {result}')
    plt.axis('off')
    plt.show()
    
    return predicted.item()

# 实施裂缝分割函数（进阶功能）
def segment_cracks(model, image_path):
    # 这里可以实现更高级的裂缝分割功能
    # 例如使用U-Net等分割模型进行像素级裂缝预测
    pass

# 主函数示例
def main():
    # 初始化模型
    model = CrackDetectionCNN()
    
    # 加载数据
    data_dir = 'path/to/your/bridge_crack_dataset'
    train_loader, val_loader = load_data(data_dir, batch_size=32)
    
    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'bridge_crack_model.pth')
    
    # 测试模型
    test_image = 'path/to/test/image.jpg'
    result = predict(trained_model, test_image)
    print(f"预测结果: {'裂缝' if result == 1 else '无裂缝'}")

if __name__ == "__main__":
    main()
