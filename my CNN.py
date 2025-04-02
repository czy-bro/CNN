import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

#超参数设置
num_epochs = 25
batch_size = 64
learning_rate = 0.001
dropout_rate = 0.5
num_conv_layers = 3
learning_rate_decay = 0.9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据预处理与加载
data_root = "D:/"
transform = transforms.Compose([
   transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
   ])

#加载数据
train_dataset = torchvision.datasets.CIFAR10(root=data_root,train=True,download=False,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_root,train=False,download=False,transform=transform)

#训练集划分为训练集和验证集
train_size = int(0.9*len(train_dataset))
val_size = len(train_dataset) - train_size
train_set,val_set = random_split(train_dataset,[train_size,val_size])

train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True,num_workers= 2)
val_loader = DataLoader(val_set,batch_size = batch_size,shuffle = False,num_workers= 2)
test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False,num_workers= 2)
#？shuffle


#定义CNN结构


class CNN(nn.Module):
    def __init__(self,num_conv_layers, dropout_rate):
        super().__init__()
        in_channels = 3
        out_channels = 32
        kernel_size = 3
        layers = []
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=1,stride=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            in_channels = out_channels
            out_channels = 2*out_channels
        self.model = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        final_size = 32 // (2 ** num_conv_layers)
        self.fc1 = nn.Linear(final_size * final_size * (out_channels // 2), 256)
        self.fc2 = nn.Linear(256,10)


    def forward(self,x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x= F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#训练与验证
def train_and_evaluate(model,num_epochs,learning_rate,learning_rate_decay):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = learning_rate_decay)

    train_losses = []
    val_losses = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        #训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images,labels in train_loader:
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs= model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss/len(train_loader)
        train_acc = correct/total
        train_losses.append(train_loss)

        #验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct/total
        val_losses.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),"best_cnn_model.pth")

        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch[{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


    #画loss变化曲线
    plt.plot(range(num_epochs),train_losses,label = "train loss")
    plt.plot(range(num_epochs),val_losses,label = "validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss curve")
    plt.show()

    torch.save(model.state_dict(), "final_cnn_model.pth")
    print("最终模型已保存")

    # train.py
if __name__ == "__main__":
    model = CNN(num_conv_layers=num_conv_layers, dropout_rate=dropout_rate)
    train_and_evaluate(model, num_epochs, learning_rate, learning_rate_decay)

    #测试模型

    model.load_state_dict(torch.load("best_cnn_model.pth"))  # ✅ 加载最优模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total  # 计算测试集准确率
    print(f"测试集准确率: {test_acc:.4f}")





