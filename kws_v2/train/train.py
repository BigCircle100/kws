import torch
from torch import nn
from model import Cnn6

import os
from dataset import SyntheticVoiceDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

root_dir = "dataset/"
model_save = f"model_weight/"
os.makedirs(model_save, exist_ok=True)

sample_rate = 8000
learning_rate = 0.001
num_epochs = 5
batch_size = 32
num_classes = 5
start_epoch = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Cnn6(num_classes, sample_rate).to(device)  
# model.load_state_dict(torch.load(f"model_weight/model_weights_5.pth")) # 从某个权重继续训练，如果不需要就注释掉
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = SyntheticVoiceDataset(root_dir=root_dir+'/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
val_dataset = SyntheticVoiceDataset(root_dir=root_dir+'/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(start_epoch, start_epoch+num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for audio, labels in train_loader:
        audio, labels = audio.to(device), labels.to(device)  
        optimizer.zero_grad()  
        outputs = model(audio)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 验证
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)  
            outputs = model(audio)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{start_epoch+num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    # 保存模型权重
    torch.save(model.state_dict(), f'{model_save}/model_weights_{epoch+1}.pth')

# 绘制图片
plt.figure(figsize=(12, 5))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率图
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(f'training_curves.png', dpi=300, bbox_inches='tight')  
