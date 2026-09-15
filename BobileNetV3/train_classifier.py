import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. นิยาม Transformation สำหรับเตรียมรูปภาพ
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. โหลด Dataset
image_datasets = {x: datasets.ImageFolder(f'dataset/{x}', data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'val']}

# 3. โหลด Pre-trained MobileNetV3 Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# เปลี่ยน Output Layer ให้จำแนกเหลือ 2 คลาส (non_road: 0, road: 1)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. เริ่มต้นเทรน (15 Epochs)
print("Starting Training...")
for epoch in range(15):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # --- จุดที่ต้องเพิ่ม: คำนวณ Loss เฉลี่ยประจำ Epoch ---
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'Epoch {epoch+1}/15 [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
# 5. บันทึก Weights ของโมเดลที่เทรนเสร็จแล้ว
torch.save(model.state_dict(), 'road_classifier.pth')
print("Model saved as 'road_classifier.pth'")