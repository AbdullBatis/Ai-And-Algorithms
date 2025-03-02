
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import csv
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) 
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load data
class nMNIST_Dataset(data.Dataset):
    def __init__(self, csv_file_x, csv_file_y, size):
        super().__init__()
        self.size = size
        self.data, self.label = self.load_data(csv_file_x, csv_file_y)
        self.T = transforms.ToTensor()

    def load_data(self, csv_file_x, csv_file_y):
        img_data = np.zeros((self.size, 28, 28))
        label_data = np.zeros((self.size, 10))
        with open(csv_file_x, 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            for i, data in enumerate(csvreader):
                img = np.array(data, dtype='int64')
                img_data[i] = img.reshape((28, 28))
        with open(csv_file_y, 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            for i, data in enumerate(csvreader):
                label = np.array(data, dtype='int64')
                label_data[i] = label
        return img_data, label_data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        data_point = self.T(data_point).to(torch.float32)
        data_label = torch.tensor(data_label).argmax()
        return data_point, data_label

# File paths (adjust as needed)
trainx_file = './n-MNIST/trainx.csv'
trainy_file = './n-MNIST/trainy.csv'
testx_file = './n-MNIST/testx.csv'
testy_file = './n-MNIST/testy.csv'

train_dataset = nMNIST_Dataset(trainx_file, trainy_file, size=60000)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = nMNIST_Dataset(testx_file, testy_file, size=10000)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Function to train the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Function to test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Main script to run the experiments with the additional convolutional layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, 15):
    train(model, device, train_loader, optimizer, epoch)
accuracy = test(model, device, test_loader)

# Print results
print(f"Accuracy with additional convolutional layer: {accuracy:.2f}%")
